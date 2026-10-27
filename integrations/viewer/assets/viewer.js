"use strict";
(() => {
  const el = id => document.getElementById(id);
  const status = el("render-status");
  try {
    const scenes = JSON.parse(el("scene-data").textContent);
    const canvas = el("viewport");
    const gl = canvas.getContext("webgl", {alpha: true, antialias: true});
    if (!gl) throw new Error("此浏览器未提供 WebGL；几何尚未显示。请用启用 WebGL 的浏览器打开此文件。");
    const shader = (type, source) => {
      const s = gl.createShader(type); gl.shaderSource(s, source); gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
      return s;
    };
    const program = gl.createProgram();
    gl.attachShader(program, shader(gl.VERTEX_SHADER, `
      attribute vec3 position; attribute vec3 color; attribute vec3 normal;
      uniform mat3 rotation; uniform vec2 fit; uniform float zoom; uniform float size;
      uniform bool lit; varying vec3 rgb;
      void main(){
        vec3 p = rotation * position;
        gl_Position=vec4(p.xy*fit*zoom*0.50, p.z*0.22, 1.0);
        gl_PointSize=size;
        float light=lit ? 0.4+0.6*abs(dot(normalize(rotation*normal),normalize(vec3(0.3,0.6,1.0)))) : 1.0;
        rgb=color*light;
      }`));
    gl.attachShader(program, shader(gl.FRAGMENT_SHADER, `
      precision mediump float; varying vec3 rgb; uniform bool dots;
      void main(){if(dots && distance(gl_PointCoord,vec2(0.5))>0.5) discard;gl_FragColor=vec4(rgb,1.0);}`));
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(program));
    gl.useProgram(program); gl.enable(gl.DEPTH_TEST); gl.clearColor(0,0,0,0);
    const attrs = Object.fromEntries(["position","color","normal"].map(n=>[n,gl.getAttribLocation(program,n)]));
    const uniforms = Object.fromEntries(["rotation","fit","zoom","size","lit","dots"].map(n=>[n,gl.getUniformLocation(program,n)]));
    let scene, buffers={}, selected=null, yaw=0, pitch=0, zoom=1, drag=null, renderNumber=0;
    const pointLimits = gl.getParameter(gl.ALIASED_POINT_SIZE_RANGE);
    const createBuffer = (positions,colors,normals=[]) => {
      if (!normals.length) normals=positions.map((_,i)=>i%3===2 ? 1 : 0);
      const obj={count:positions.length/3};
      for(const [key,values] of Object.entries({position:positions,color:colors,normal:normals})){
        const data=new Float32Array(values);
        if(!data.every(Number.isFinite)) throw new Error("显示坐标无法转为有限 float32");
        const b=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,b);gl.bufferData(gl.ARRAY_BUFFER,data,gl.STATIC_DRAW);obj[key]=b;
      } return obj;
    };
    const bind = buffer => {for(const key of Object.keys(attrs)){gl.bindBuffer(gl.ARRAY_BUFFER,buffer[key]);gl.vertexAttribPointer(attrs[key],3,gl.FLOAT,false,0,0);gl.enableVertexAttribArray(attrs[key]);}};
    const discardBuffers=()=>{for(const b of Object.values(buffers))for(const key of Object.keys(attrs))gl.deleteBuffer(b[key]);buffers={};};
    function rotation(){
      const c=Math.cos(yaw),s=Math.sin(yaw),a=Math.cos(pitch),b=Math.sin(pitch);
      // Column-major Rx(pitch)*Ry(yaw), same scale on all three world axes.
      return new Float32Array([c,b*s,-a*s, 0,a,b, s,-b*c,a*c]);
    }
    function draw(){
      const dpr=Math.min(window.devicePixelRatio||1,2);
      const w=Math.max(1,Math.round(canvas.clientWidth*dpr)),h=Math.max(1,Math.round(canvas.clientHeight*dpr));
      if(canvas.width!==w||canvas.height!==h){canvas.width=w;canvas.height=h;}
      gl.viewport(0,0,w,h);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);
      const aspect=w/h;
      gl.uniformMatrix3fv(uniforms.rotation,false,rotation());
      gl.uniform2f(uniforms.fit,aspect>=1?1/aspect:1,aspect>=1?1:aspect);
      gl.uniform1f(uniforms.zoom,zoom);
      gl.uniform1f(uniforms.size,Math.max(pointLimits[0],Math.min(pointLimits[1],Number(el("point-size").value)*dpr)));
      const mode=el("mode").value;
      const b=mode==="surface"?buffers.surface:mode==="wire"?buffers.wire:buffers.points;
      gl.uniform1i(uniforms.lit,mode==="surface");gl.uniform1i(uniforms.dots,mode==="points");
      bind(b);gl.drawArrays(mode==="surface"?gl.TRIANGLES:mode==="wire"?gl.LINES:gl.POINTS,0,b.count);
      gl.uniform1i(uniforms.lit,0);gl.uniform1i(uniforms.dots,0);
      bind(buffers.axes);gl.drawArrays(gl.LINES,0,buffers.axes.count);
      if(selected!==null){
        gl.disable(gl.DEPTH_TEST);gl.uniform1i(uniforms.dots,1);
        gl.uniform1f(uniforms.size,Math.min(pointLimits[1],12*dpr));
        bind(buffers.selected);gl.drawArrays(gl.POINTS,0,1);gl.enable(gl.DEPTH_TEST);
      }
      const error=gl.getError();if(error!==gl.NO_ERROR)throw new Error("WebGL 绘制失败，错误码 "+error);
      renderNumber++;
      el("camera-state").textContent=`方位 ${(yaw*180/Math.PI).toFixed(0)}° · 俯仰 ${(pitch*180/Math.PI).toFixed(0)}° · ${zoom.toFixed(2)}×`;
      status.textContent=`已绘制 · ${scene.vertices.length.toLocaleString()} 顶点 · ${scene.faces.length.toLocaleString()} 三角面 · ${el("mode").selectedOptions[0].textContent}`;
      status.dataset.renderCount=String(renderNumber);status.dataset.scene=el("scene").value;status.dataset.mode=mode;status.dataset.state="rendered";
    }
    const normalized=p=>p.map((v,i)=>(v-scene.center[i])*scene.view_scale);
    const color=i=>scene.colors?scene.colors[i].map(x=>x/255):[0.37,0.79,0.77];
    function loadScene(){
      discardBuffers();scene=scenes[Number(el("scene").value)];selected=null;yaw=0;pitch=0;zoom=1;
      const points=scene.vertices.map(normalized),colors=scene.vertices.map((_,i)=>color(i));
      buffers.points=createBuffer(points.flat(),colors.flat());
      const facePositions=[],faceColors=[],normals=[],wirePositions=[],wireColors=[];
      for(const f of scene.faces){
        const [a,b,c]=f.map(i=>points[i]),u=b.map((v,i)=>v-a[i]),v=c.map((x,i)=>x-a[i]);
        let n=[u[1]*v[2]-u[2]*v[1],u[2]*v[0]-u[0]*v[2],u[0]*v[1]-u[1]*v[0]];
        const len=Math.hypot(...n);if(!len)throw new Error("退化三角面不能显示为有效表面");n=n.map(x=>x/len);
        for(const i of f){facePositions.push(...points[i]);faceColors.push(...colors[i]);normals.push(...n);}
        for(const [i,j] of [[f[0],f[1]],[f[1],f[2]],[f[2],f[0]]]){wirePositions.push(...points[i],...points[j]);wireColors.push(...colors[i],...colors[j]);}
      }
      buffers.surface=createBuffer(facePositions,faceColors,normals);buffers.wire=createBuffer(wirePositions,wireColors);
      // An orientation guide centered on the display bbox, not a source-world origin.
      buffers.axes=createBuffer([0,0,0,1.25,0,0,0,0,0,0,1.25,0,0,0,0,0,0,1.25],[1,.35,.35,1,.35,.35,.35,1,.55,.35,1,.55,.4,.65,1,.4,.65,1]);
      el("scene-title").textContent=scene.title;el("note").textContent=scene.note;
      const badges={real_photos:"真实照片来源声明",synthetic:"合成数据 · 非真实物体",unspecified:"来源未声明"};
      el("badge").textContent=badges[scene.data_kind];el("badge").className="badge"+(scene.data_kind==="synthetic"?" synthetic":"");
      const facts=[["顶点",String(scene.vertices.length)],["三角面",String(scene.faces.length)],["颜色",scene.colors?"原始 RGB":"统一预览色"],["单位声明",scene.declared_unit||"未知 / 任意尺度"],["输入格式",scene.format]];
      el("facts").replaceChildren();for(const [k,v] of facts){const dt=document.createElement("dt"),dd=document.createElement("dd");dt.textContent=k;dd.textContent=v;el("facts").append(dt,dd);}
      el("geometry-label").textContent=scene.faces.length?"三角网格预览 · 非 STEP 导入":"稀疏点云 · 非完整表面";
      el("provenance").textContent=JSON.stringify({source:scene.source_path,sha256:scene.sha256,bbox:scene.bbox,evidence:scene.evidence,note:"证据文件仅绑定哈希，未在本页重新验收；轴线位于显示包围盒中心。坐标显示保留原始 double，绘制使用归一化 float32。"},null,2);
      for(const option of el("mode").options)option.disabled=option.value!=="points"&&!scene.faces.length;
      el("mode").value=scene.faces.length?"surface":"points";
      el("vertex-index").value="0";el("vertex-index").max=String(scene.vertices.length-1);el("vertex-info").textContent="尚未选择顶点";
      draw();
    }
    const guarded=fn=>(...args)=>{try{fn(...args);}catch(error){status.textContent="显示失败："+error.message;status.dataset.state="error";console.error(error);}};
    for(const [i,s] of scenes.entries()){const option=document.createElement("option");option.value=String(i);option.textContent=s.title;el("scene").append(option);}
    el("scene").addEventListener("change",guarded(loadScene));el("mode").addEventListener("change",guarded(draw));
    const change=(dy,dp,dz=1)=>{yaw+=dy;pitch=Math.max(-Math.PI/2,Math.min(Math.PI/2,pitch+dp));zoom=Math.max(.2,Math.min(8,zoom*dz));draw();};
    for(const [id,values] of Object.entries({left:[-.3,0],right:[.3,0],up:[0,-.25],down:[0,.25],"zoom-in":[0,0,1.2],"zoom-out":[0,0,1/1.2]}))el(id).addEventListener("click",guarded(()=>change(...values)));
    el("reset").addEventListener("click",guarded(()=>{yaw=0;pitch=0;zoom=1;draw();}));
    el("point-size").addEventListener("input",guarded(()=>{el("point-size-value").textContent=el("point-size").value;draw();}));
    el("inspect").addEventListener("click",guarded(()=>{
      const text=el("vertex-index").value;const i=Number(text);
      if(!text||!Number.isInteger(i)||i<0||i>=scene.vertices.length){el("vertex-info").textContent="索引超出范围";return;}
      if(buffers.selected)for(const k of Object.keys(attrs))gl.deleteBuffer(buffers.selected[k]);
      selected=i;buffers.selected=createBuffer(normalized(scene.vertices[i]),[1,.85,.3]);
      el("vertex-info").textContent=`行 ${i}\nX ${scene.vertices[i][0]}\nY ${scene.vertices[i][1]}\nZ ${scene.vertices[i][2]}\n单位声明：${scene.declared_unit||"未知"}（黄色标记始终置前）`;draw();
    }));
    canvas.addEventListener("pointerdown",e=>{drag=[e.clientX,e.clientY];canvas.setPointerCapture(e.pointerId);});
    canvas.addEventListener("pointermove",guarded(e=>{if(drag){change((e.clientX-drag[0])*.008,(e.clientY-drag[1])*.008);drag=[e.clientX,e.clientY];}}));
    for(const name of ["pointerup","pointercancel","lostpointercapture"])canvas.addEventListener(name,()=>{drag=null;});
    canvas.addEventListener("wheel",guarded(e=>{e.preventDefault();change(0,0,Math.exp(Math.max(-1,Math.min(1,-e.deltaY*.001))));}),{passive:false});
    canvas.addEventListener("keydown",guarded(e=>{const map={ArrowLeft:[-.15,0],ArrowRight:[.15,0],ArrowUp:[0,-.15],ArrowDown:[0,.15]};if(map[e.key]){e.preventDefault();change(...map[e.key]);}}));
    canvas.addEventListener("webglcontextlost",e=>{e.preventDefault();status.textContent="显示上下文已丢失，请重新打开页面。";status.dataset.state="error";});
    new ResizeObserver(guarded(draw)).observe(canvas);loadScene();
  } catch(error) {status.textContent="显示失败："+error.message;status.dataset.state="error";console.error(error);}
})();
