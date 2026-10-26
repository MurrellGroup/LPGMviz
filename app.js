import * as THREE from './vendor/three.module.min.js';

const $ = id => document.getElementById(id);
const canvas = $('viewport');
const renderer = new THREE.WebGLRenderer({canvas, antialias:true, alpha:false, preserveDrawingBuffer:true});
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.08;
renderer.setClearColor(0xffffff);
renderer.setPixelRatio(Math.min(devicePixelRatio,2));

const sphere = new THREE.SphereGeometry(1,28,20);
const cylinder = new THREE.CylinderGeometry(1,1,1,16);
const cone = new THREE.ConeGeometry(1,1,18);
const up = new THREE.Vector3(0,1,0);
const direction = new THREE.Vector3();
const midpoint = new THREE.Vector3();
const materialCache = new Map();
const scales = [.72,1.05,1,.96,.92,.86,1.12,1.08,1.08,1.04,1.14,1.2,1.22,.68];
const originalOrientation = new THREE.Quaternion().setFromEuler(new THREE.Euler(.38,-.55,0,'YXZ'));
const PLAYBACK_SECONDS = 32;
const BOND_FADE_START = .95;
const state = {manifest:null, method:'fragment', sample:0, t:0, playing:false,
  ready:false, views:[], cache:new Map(), serial:0, orientation:originalOrientation.clone(), zoom:1};
const query = new URLSearchParams(location.search);
if(query.get('method')==='bm') state.method='bm';
state.playing=query.get('autoplay')==='1' && !matchMedia('(prefers-reduced-motion: reduce)').matches;

function material(fragment,bond=false,cross=false) {
  const key = `${fragment}-${bond}-${cross}`;
  if(!materialCache.has(key)) materialCache.set(key,new THREE.MeshStandardMaterial({
    color:new THREE.Color().setStyle(`hsl(${(fragment*137.507764+28)%360},76%,58%)`),
    roughness:bond?.32:.26,metalness:bond?.04:.025,
    transparent:cross,opacity:cross?0:1,depthWrite:!cross}));
  return materialCache.get(key);
}
const arrowMaterial = new THREE.MeshStandardMaterial({color:'#464b52',roughness:.5,metalness:0});

function makeScene() {
  const scene = new THREE.Scene();scene.background=new THREE.Color('#ffffff');
  scene.add(new THREE.HemisphereLight(0xffffff,0x263630,2.25));
  for(const [color,intensity,xyz] of [[0xffffff,3.1,[-4,6,9]],[0x9bc9ff,1.35,[7,-2,5]],[0xffd6a1,1.15,[-6,-5,-3]]]) {
    const light = new THREE.DirectionalLight(color,intensity);light.position.set(...xyz);scene.add(light);
  }
  const root = new THREE.Group();scene.add(root);
  const camera = new THREE.OrthographicCamera(-1,1,1,-1,.01,2000);
  camera.position.set(0,0,1000);camera.lookAt(0,0,0);
  return {scene,root,camera};
}

async function array(path, Type=Float32Array) {
  const response = await fetch(path);
  if(!response.ok) throw new Error(`Unable to load ${path}`);
  return new Type(await response.arrayBuffer());
}
async function loadSample(index) {
  if(state.cache.has(index)) return state.cache.get(index);
  const record=state.manifest.samples[index];
  const promise=(async()=>{
    const data={};
    await Promise.all(['fragment','bm'].flatMap(method=>['conditional','generated'].map(async kind=>{
      const meta=record.methods[method][kind];
      const [positions,centers,rotations]=await Promise.all([
        array(meta.positions),meta.centers?array(meta.centers):null,meta.rotations?array(meta.rotations):null]);
      if(positions.length!==meta.frames*record.atom_count*3) throw new Error('Invalid trajectory length');
      data[`${method}-${kind}`]={meta,positions,centers,rotations};
    })));
    return {record,data};
  })();
  state.cache.set(index,promise);
  try{return await promise;}catch(error){state.cache.delete(index);throw error;}
}

function createView(record,data,kind) {
  const view={...makeScene(),record,...data,kind,frame:0,frameNext:0,mix:0,atoms:[],bonds:[],arrows:[]};
  for(let atom=0;atom<record.atom_count;atom++) {
    const mesh=new THREE.Mesh(sphere,material(record.fragment_ids[atom]));
    mesh.scale.setScalar(.364*(scales[data.meta.final_labels[atom]]||1));
    view.root.add(mesh);view.atoms.push(mesh);
  }
  for(const [a,b] of data.meta.bonds) {
    const cross=record.fragment_ids[a]!==record.fragment_ids[b];
    const first=new THREE.Mesh(cylinder,material(record.fragment_ids[a],true,cross));
    const second=new THREE.Mesh(cylinder,material(record.fragment_ids[b],true,cross));
    view.root.add(first,second);view.bonds.push({a,b,cross,first,second});
  }
  // Convert the recorded SO(3) matrices once. Inter-frame display uses shortest-arc
  // SLERP, retaining orthogonal axes; exact recorded times use the raw matrices.
  if(data.rotations) {
    view.quaternions=new Float64Array(data.meta.frames*record.fragment_count*4);
    const matrix=new THREE.Matrix4(),quaternion=new THREE.Quaternion();
    for(let i=0;i<data.meta.frames*record.fragment_count;i++) {
      const o=i*9,r=data.rotations;
      matrix.set(r[o],r[o+3],r[o+6],0,r[o+1],r[o+4],r[o+7],0,r[o+2],r[o+5],r[o+8],0,0,0,0,1);
      quaternion.setFromRotationMatrix(matrix).normalize().toArray(view.quaternions,i*4);
    }
  }
  if(data.rotations) for(let f=0;f<record.fragment_count;f++) for(let axis=0;axis<3;axis++) {
    const shaft=new THREE.Mesh(cylinder,arrowMaterial),tip=new THREE.Mesh(cone,arrowMaterial);
    view.root.add(shaft,tip);view.arrows.push({fragment:f,axis,shaft,tip});
  }
  return view;
}

// Display interpolation between adjacent saved physical times. No temporal
// averaging, denoising, curve fitting, overshoot, or changes to the source arrays.
function frameBracket(times,t) {
  let lo=0,hi=times.length-1;
  while(lo<hi){const mid=(lo+hi)>>1;if(times[mid]<t)lo=mid+1;else hi=mid;}
  if(lo===0 || times[lo]===t)return {left:lo,right:lo,mix:0};
  return {left:lo-1,right:lo,mix:(t-times[lo-1])/(times[lo]-times[lo-1])};
}
function bondOpacity(t) {
  const x=Math.max(0,Math.min(1,(t-BOND_FADE_START)/(1-BOND_FADE_START)));
  return x*x*(3-2*x);
}
function place(mesh,start,stop,radius) {
  direction.subVectors(stop,start);const length=direction.length();
  mesh.visible=length>1e-8;if(!mesh.visible)return;
  mesh.position.copy(start).add(stop).multiplyScalar(.5);
  mesh.quaternion.setFromUnitVectors(up,direction.multiplyScalar(1/length));
  mesh.scale.set(radius,length,radius);
}
const origin=new THREE.Vector3(),axisVector=new THREE.Vector3(),shaftEnd=new THREE.Vector3(),arrowEnd=new THREE.Vector3();
const nextPoint=new THREE.Vector3(),rotationLeft=new THREE.Quaternion(),rotationRight=new THREE.Quaternion();
function updateView(view) {
  const {record,meta}=view;
  const {left,right,mix}=frameBracket(meta.times,state.t);
  view.frame=left;view.frameNext=right;view.mix=mix;
  const offset=left*record.atom_count*3,nextOffset=right*record.atom_count*3;
  for(let i=0;i<record.atom_count;i++) {
    const p=view.atoms[i].position.fromArray(view.positions,offset+3*i);
    if(mix>0)p.lerp(nextPoint.fromArray(view.positions,nextOffset+3*i),mix);
  }
  const crossOpacity=bondOpacity(state.t);
  for(const b of view.bonds) {
    const start=view.atoms[b.a].position,end=view.atoms[b.b].position;
    midpoint.copy(start).add(end).multiplyScalar(.5);
    place(b.first,start,midpoint,.105);place(b.second,midpoint,end,.105);
    if(b.cross) {
      b.first.material.opacity=crossOpacity;b.second.material.opacity=crossOpacity;
      b.first.visible=b.first.visible && crossOpacity>0;b.second.visible=b.second.visible && crossOpacity>0;
    }
  }
  for(const arrow of view.arrows) {
    const f=arrow.fragment;
    origin.fromArray(view.centers,(left*record.fragment_count+f)*3);
    if(mix>0)origin.lerp(nextPoint.fromArray(view.centers,(right*record.fragment_count+f)*3),mix);
    if(mix===0)axisVector.fromArray(view.rotations,((left*record.fragment_count+f)*3+arrow.axis)*3);
    else {
      rotationLeft.fromArray(view.quaternions,(left*record.fragment_count+f)*4);
      rotationRight.fromArray(view.quaternions,(right*record.fragment_count+f)*4);
      rotationLeft.slerp(rotationRight,mix);
      axisVector.set(arrow.axis===0?1:0,arrow.axis===1?1:0,arrow.axis===2?1:0).applyQuaternion(rotationLeft);
    }
    // Actual recorded latent axes at the knots; interpolated rotations in between.
    shaftEnd.copy(origin).addScaledVector(axisVector,1.8);
    arrowEnd.copy(origin).addScaledVector(axisVector,2.15);
    place(arrow.shaft,origin,shaftEnd,.034);
    place(arrow.tip,shaftEnd,arrowEnd,.115);
  }
  view.root.quaternion.copy(state.orientation);
}

function render() {
  if(!state.ready)return;
  const width=canvas.clientWidth,height=canvas.clientHeight;
  if(canvas.width!==Math.round(width*renderer.getPixelRatio()) || canvas.height!==Math.round(height*renderer.getPixelRatio()))renderer.setSize(width,height,false);
  renderer.setScissorTest(false);renderer.clear();renderer.setScissorTest(true);
  const half=width/2;
  for(let i=0;i<state.views.length;i++) {
    const view=state.views[i];updateView(view);
    const aspect=half/height;
    const [rx,ry]=view.record.display_half_extent;
    const visibleHeight=2*Math.max(ry,rx/aspect)/state.zoom;
    view.camera.left=-visibleHeight*aspect/2;view.camera.right=visibleHeight*aspect/2;
    view.camera.top=visibleHeight/2;view.camera.bottom=-visibleHeight/2;view.camera.updateProjectionMatrix();
    renderer.setViewport(i*half,0,half,height);renderer.setScissor(i*half,0,half,height);
    renderer.render(view.scene,view.camera);
  }
  renderer.setScissorTest(false);
  $('timeline').value=String(state.t);$('time').textContent=`t = ${state.t.toFixed(3)}`;
}

function playbackUI() {
  $('pauseIcon').toggleAttribute('hidden',!state.playing);$('playIcon').toggleAttribute('hidden',state.playing);
  $('play').setAttribute('aria-label',state.playing?'Pause':state.t>=1?'Replay':'Play');
}
function remember() {
  const url=new URL(location.href);url.searchParams.set('method',state.method);
  url.searchParams.set('sample',state.manifest.samples[state.sample].id);history.replaceState({},'',url);
}
async function showSample() {
  const serial=++state.serial;state.ready=false;
  $('status').hidden=false;$('status').textContent='Loading paths…';
  $('play').disabled=true;$('timeline').disabled=true;
  try {
    const loaded=await loadSample(state.sample);
    if(serial!==state.serial)return;
    state.views=['conditional','generated'].map(kind=>createView(loaded.record,loaded.data[`${state.method}-${kind}`],kind));
    for(const button of document.querySelectorAll('[data-method]'))button.setAttribute('aria-pressed',String(button.dataset.method===state.method));
    $('sample').value=String(state.sample);
    state.ready=true;$('status').hidden=true;$('play').disabled=false;$('timeline').disabled=false;
    playbackUI();remember();render();
  }catch(error){$('status').textContent=error.message;console.error(error);}
}
function chooseSample(index) {
  state.sample=(index+state.manifest.samples.length)%state.manifest.samples.length;
  state.t=0;showSample();
}
for(const button of document.querySelectorAll('[data-method]')) button.addEventListener('click',()=>{
  if(button.dataset.method!==state.method){state.method=button.dataset.method;showSample();}
});
$('sample').addEventListener('change',()=>chooseSample(Number($('sample').value)));
$('previous').addEventListener('click',()=>chooseSample(state.sample-1));
$('next').addEventListener('click',()=>chooseSample(state.sample+1));
$('play').addEventListener('click',()=>{if(!state.playing && state.t>=1){state.t=0;render();}state.playing=!state.playing;playbackUI();});
$('timeline').addEventListener('input',()=>{state.t=Number($('timeline').value);state.playing=false;playbackUI();render();});

let drag=null;
canvas.addEventListener('pointerdown',event=>{
  if(event.button!==0)return;drag={x:event.clientX,y:event.clientY};canvas.setPointerCapture(event.pointerId);
});
canvas.addEventListener('pointermove',event=>{
  if(!drag)return;
  const dx=event.clientX-drag.x,dy=event.clientY-drag.y;
  const q=new THREE.Quaternion().setFromEuler(new THREE.Euler(dy*.007,dx*.007,0,'YXZ'));
  state.orientation.premultiply(q).normalize();drag={x:event.clientX,y:event.clientY};render();
});
canvas.addEventListener('pointerup',()=>{drag=null;});canvas.addEventListener('pointercancel',()=>{drag=null;});
canvas.addEventListener('wheel',event=>{event.preventDefault();state.zoom=Math.max(.7,Math.min(2.5,state.zoom*Math.exp(-event.deltaY*.001)));render();},{passive:false});
window.addEventListener('resize',render);
let last=performance.now();
function tick(now) {
  const elapsed=Math.min((now-last)/1000,.1);last=now;
  if(state.ready && state.playing && !document.hidden) {
    state.t=Math.min(1,state.t+elapsed/PLAYBACK_SECONDS);
    if(state.t>=1){state.playing=false;playbackUI();}
    render();
  }
  requestAnimationFrame(tick);
}
requestAnimationFrame(tick);

// Read-only numerical inspection for browser checks; no alternative render path.
window.animationAudit=()=>({ready:state.ready,method:state.method,sample:state.manifest?.samples[state.sample].id,
  t:state.t,playing:state.playing,zoom:state.zoom,orientation:state.orientation.toArray(),playbackSeconds:PLAYBACK_SECONDS,
  views:state.views.map(v=>({kind:v.kind,frame:v.frame,frameNext:v.frameNext,mix:v.mix,recordedTime:v.meta.times[v.frame],
    atoms:v.atoms.map(m=>m.position.toArray()),bonds:v.bonds.map(b=>[b.a,b.b]),
    bondDisplay:v.bonds.map(b=>({cross:b.cross,visible:b.first.visible,opacity:b.first.material.opacity})),
    colors:v.atoms.map(m=>m.material.color.getHexString()),
    arrows:v.arrows.map(a=>({fragment:a.fragment,axis:a.axis,origin:a.shaft.position.clone().addScaledVector(
      new THREE.Vector3(0,1,0).applyQuaternion(a.shaft.quaternion),-a.shaft.scale.y/2).toArray(),
      direction:new THREE.Vector3(0,1,0).applyQuaternion(a.shaft.quaternion).toArray()})),
    camera:[v.camera.left,v.camera.right,v.camera.top,v.camera.bottom]}))});
try {
  const response=await fetch('manifest.json');if(!response.ok)throw new Error('Unable to load manifest');
  state.manifest=await response.json();
  state.manifest.samples.forEach((sample,i)=>{
    const option=document.createElement('option');option.value=String(i);option.textContent=`Sample ${i+1} / ${state.manifest.samples.length}`;$('sample').append(option);
  });
  const requested=state.manifest.samples.findIndex(s=>s.id===query.get('sample'));if(requested>=0)state.sample=requested;
  await showSample();
}catch(error){$('status').textContent=error.message;console.error(error);}
