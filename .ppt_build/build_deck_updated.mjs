import fs from 'node:fs/promises';
import path from 'node:path';
import { Presentation, PresentationFile } from 'file:///C:/Users/19131/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs';

const W=1280,H=720;
const C={red:'#BC1D2D', orange:'#E0522E', navy:'#263238', gray:'#667085', light:'#F7F8FA', pale:'#FDECEC', green:'#2E8B57', white:'#FFFFFF', line:'#E5E7EB'};
const pres=Presentation.create({slideSize:{width:W,height:H}});
const outDir='E:/任务一_KV_Cache分页仿真器_完整源代码/ppt_output';
const buildDir='E:/任务一_KV_Cache分页仿真器_完整源代码/.ppt_build';
await fs.mkdir(outDir,{recursive:true});
await fs.mkdir(buildDir,{recursive:true});

function box(slide,{x,y,w,h,fill='none',line='none',text='',fs=22,color=C.navy,bold=false,align='left',valign='top',radius=null}){
  const s=slide.shapes.add({geometry: radius?'roundRect':'rect',position:{left:x,top:y,width:w,height:h},fill,line:{style:'solid',fill:line,width:line==='none'?0:1},...(radius?{borderRadius:radius}: {})});
  if(text){s.text=text;s.text.style={fontSize:fs,color,bold,align,verticalAlign:valign,fit:'shrink'};}
  return s;
}
function line(slide,x,y,w,color=C.red,width=3){return box(slide,{x,y,w,h:width,fill:color,line:color});}
function header(slide,num,title,kicker='任务一｜阶段汇报'){
  slide.background.fill=C.white;
  box(slide,{x:0,y:0,w:W,h:12,fill:C.red,line:C.red});
  box(slide,{x:60,y:38,w:320,h:22,text:kicker,fs:14,color:C.red,bold:true});
  box(slide,{x:60,y:72,w:1040,h:54,text:title,fs:32,color:C.navy,bold:true});
  box(slide,{x:1150,y:72,w:70,h:38,text:String(num).padStart(2,'0'),fs:24,color:C.red,bold:true,align:'right'});
  line(slide,60,138,1160,C.line,1);
  box(slide,{x:60,y:684,w:750,h:18,text:'东华大学 2026 专业实习｜面向大模型推理优化的操作系统级任务',fs:11,color:C.gray});
  box(slide,{x:1150,y:684,w:70,h:18,text:String(num).padStart(2,'0'),fs:11,color:C.gray,align:'right'});
}
function bullet(slide,x,y,w,title,body,accent=C.red){
  box(slide,{x,y,w:8,h:58,fill:accent,line:accent});
  box(slide,{x:x+24,y:y-2,w:w-24,h:26,text:title,fs:20,color:C.navy,bold:true});
  box(slide,{x:x+24,y:y+28,w:w-24,h:42,text:body,fs:15,color:C.gray});
}

// 1 cover
{
 const s=pres.slides.add();s.background.fill=C.white;
 box(s,{x:0,y:0,w:W,h:18,fill:C.red,line:C.red});
 box(s,{x:0,y:18,w:W,h:8,fill:C.orange,line:C.orange});
 box(s,{x:70,y:110,w:300,h:28,text:'东华大学｜专业实习',fs:16,color:C.red,bold:true});
 box(s,{x:70,y:178,w:900,h:120,text:'KV Cache 分页内存管理子系统',fs:44,color:C.navy,bold:true});
 box(s,{x:70,y:310,w:830,h:46,text:'对标 vLLM PagedAttention 的操作系统级任务',fs:25,color:C.gray});
 line(s,70,390,190,C.red,5);
 box(s,{x:70,y:430,w:550,h:34,text:'正式 CPU 仿真结果｜约 5 分钟',fs:20,color:C.navy,bold:true});
 // simple editable concept strip
 box(s,{x:790,y:172,w:380,h:330,fill:C.pale,line:C.pale,radius:'rounded-xl'});
 box(s,{x:835,y:220,w:290,h:42,text:'逻辑 token 块',fs:19,color:C.red,bold:true,align:'center',valign:'center'});
 box(s,{x:835,y:292,w:290,h:42,text:'页表映射',fs:19,color:C.orange,bold:true,align:'center',valign:'center'});
 box(s,{x:835,y:364,w:290,h:42,text:'物理 KV 块池',fs:19,color:C.green,bold:true,align:'center',valign:'center'});
 box(s,{x:835,y:436,w:290,h:30,text:'按需分配｜共享｜换出',fs:16,color:C.gray,align:'center'});
}
// 2 goal
{
 const s=pres.slides.add();header(s,2,'目标与系统设计','任务一｜为什么做');
 box(s,{x:70,y:170,w:520,h:90,text:'把 KV Cache 当作操作系统虚拟内存来管理',fs:28,color:C.navy,bold:true});
 box(s,{x:70,y:278,w:520,h:100,text:'请求长度不确定、并发高、显存有限。传统方案按最大上下文连续预留，造成大量内部碎片和 OOM。',fs:19,color:C.gray});
 bullet(s,680,172,470,'核心目标','固定大小 block + 页表映射，按实际生成进度占用显存');
 bullet(s,680,272,470,'OS 映射','物理页框→物理 KV 块；虚拟页→逻辑 token 块');
 bullet(s,680,372,470,'扩展能力','前缀共享 / COW；显存紧张时页面置换');
 box(s,{x:70,y:455,w:1080,h:135,fill:C.light,line:C.line,radius:'rounded-xl'});
 box(s,{x:100,y:480,w:250,h:28,text:'系统路径',fs:17,color:C.red,bold:true});
 box(s,{x:100,y:520,w:1000,h:40,text:'工作负载生成  →  离散事件仿真  →  分页 / 朴素分配  →  OOM、利用率、并发指标',fs:21,color:C.navy,bold:true});
}
// 3 completed
{
 const s=pres.slides.add();header(s,3,'已完成工作','任务一｜当前实现');
 bullet(s,70,168,520,'分页分配器','16 token/block；支持逐 token 扩页、页表地址转换、前缀共享和写时复制。');
 bullet(s,70,270,520,'对照基线','NaiveContiguousAllocator 按 M=2048 预留整段空间，用来呈现最大长度预留造成的内部浪费。');
 bullet(s,70,372,520,'运行时仿真','800 条泊松到达请求；prompt 占总长度 50%。逐步扩页记录运行中 OOM。');
 bullet(s,70,474,520,'实验输出','合成与 ShareGPT 长度轨迹对比；导出容量、碎片和 OOM 图表。');
 box(s,{x:680,y:178,w:470,h:390,fill:C.pale,line:C.pale,radius:'rounded-xl'});
 box(s,{x:720,y:210,w:390,h:30,text:'模块结构',fs:20,color:C.red,bold:true,align:'center'});
 const rows=[['workload.py','生成高并发负载'],['allocator.py','分页与朴素分配'],['simulator.py','离散事件引擎'],['experiments.py','容量扫描与评测']];
 rows.forEach((r,i)=>{const y=270+i*65; box(s,{x:725,y,w:155,h:40,fill:C.white,line:C.line,text:r[0],fs:16,color:C.navy,bold:true,align:'center',valign:'center',radius:'rounded-lg'}); box(s,{x:910,y,w:200,h:40,text:r[1],fs:16,color:C.gray,valign:'center'});});
}
// 4 results chart
{
 const s=pres.slides.add();header(s,4,'实验结果：分页显著降低显存压力','任务一｜量化对比');
 box(s,{x:70,y:170,w:510,h:70,text:'同一份 800 条请求负载',fs:24,color:C.navy,bold:true});
 box(s,{x:70,y:245,w:510,h:58,text:'总 KV 长度 128–1024 token，prompt 占 50%；M=2048，block size=16。',fs:16,color:C.gray});
 box(s,{x:660,y:170,w:500,h:300,fill:C.light,line:C.line,radius:'rounded-xl'}); box(s,{x:740,y:215,w:150,h:210,fill:C.red,line:C.red,radius:'rounded-lg'}); box(s,{x:950,y:365,w:150,h:60,fill:C.green,line:C.green,radius:'rounded-lg'}); box(s,{x:715,y:440,w:200,h:28,text:'朴素预留 102400',fs:16,color:C.navy,bold:true,align:'center'}); box(s,{x:920,y:440,w:210,h:28,text:'分页按实际 29545',fs:16,color:C.navy,bold:true,align:'center'}); box(s,{x:720,y:185,w:360,h:24,text:'总 KV 块需求（相对高度）',fs:16,color:C.gray,bold:true,align:'center'});
 box(s,{x:70,y:360,w:520,h:80,fill:C.pale,line:C.pale,radius:'rounded-xl',text:'累计预留块需求减少 71.1%\n内部未用槽比例：71.5% → 1.3%',fs:22,color:C.red,bold:true,align:'center',valign:'center'});
 box(s,{x:70,y:470,w:520,h:90,fill:C.light,line:C.line,radius:'rounded-xl',text:'零 OOM 扫描容量\n连续 20,000｜一次性 6,000｜逐步 6,000',fs:20,color:C.navy,bold:true,align:'center',valign:'center'});
 box(s,{x:660,y:500,w:500,h:55,text:'3000 块：OOM 76.5% / 21.0% / 8.6%\n仅为 CPU 仿真，尚无 GPU 实测',fs:19,color:C.green,bold:true,align:'center'});
}
// 5 issues
{
 const s=pres.slides.add();header(s,5,'已验证机制与剩余边界','任务一｜重点');
 bullet(s,70,165,520,'共享与 COW','块引用计数支撑共享释放；分支写入部分末块时分配新块。');
 bullet(s,70,270,520,'模拟 LRU 换出','按访问顺序选择块，访问换出块触发换入；CPU 侧仅保存元数据。');
 bullet(s,70,375,520,'逐 token 扩页','跨 16-token 边界才分配新块；记录中途扩页失败。');
 bullet(s,70,480,520,'验证现状','33 个单测通过，仿真器全包行覆盖 76.2%；T4 单批验证通过。');
 box(s,{x:680,y:180,w:470,h:355,fill:C.light,line:C.line,radius:'rounded-xl'});
 box(s,{x:720,y:210,w:390,h:30,text:'下一阶段需完成',fs:20,color:C.red,bold:true,align:'center'});
 const fixes=['1 完成三轮 GPU A/B','2 扩展长度档与批量扫描','3 核对原始日志及结果','4 补充实测图表与演示录屏'];
 fixes.forEach((t,i)=>{const y=270+i*58; box(s,{x:735,y,w:350,h:38,text:t,fs:17,color:C.navy,bold:i<2,valign:'center'});});
}
// 6 plan
{
 const s=pres.slides.add();header(s,6,'GPU A/B 验证路径','任务一｜下一阶段');
 const weeks=[['准备','环境','选 GPU、模型与精度；固定版本'],['基线','StaticCache','同模型运行固定容量缓存'],['分页','vLLM','16-token 块；记录引擎配置'],['负载','同一轨迹','扫描输入长度、并发和预算'],['记录','原始数据','吞吐、显存、失败类别与日志'],['交付','补充报告','写清引擎差异并录制演示']];
 weeks.forEach((r,i)=>{const x=70+(i%3)*380,y=175+Math.floor(i/3)*205; box(s,{x,y,w:330,h:155,fill:i<3?C.pale:C.light,line:i<3?C.pale:C.line,radius:'rounded-xl'}); box(s,{x:x+22,y:y+20,w:80,h:24,text:r[0],fs:15,color:C.red,bold:true}); box(s,{x:x+22,y:y+55,w:280,h:30,text:r[1],fs:22,color:C.navy,bold:true}); box(s,{x:x+22,y:y+92,w:280,h:48,text:r[2],fs:15,color:C.gray});});
}
// 7 close
{
 const s=pres.slides.add();header(s,7,'阶段结论与答辩重点','任务一｜汇报收束');
 box(s,{x:70,y:180,w:1080,h:95,fill:C.pale,line:C.pale,radius:'rounded-xl',text:'CPU 仿真已复现分配差异；真实 GPU 收益仍待实测。',fs:28,color:C.red,bold:true,align:'center',valign:'center'});
 box(s,{x:70,y:330,w:500,h:170,fill:C.light,line:C.line,radius:'rounded-xl'});
 box(s,{x:100,y:360,w:440,h:30,text:'目前能回答的问题',fs:20,color:C.navy,bold:true});
 box(s,{x:100,y:405,w:420,h:86,text:'为什么分页有效？\n实际长度远小于最大长度时，按需分块能减少内部浪费。\n同样容量下，为什么能接纳更多请求？\n因为每个请求只占用已生成部分的 KV 块。',fs:16,color:C.gray});
 box(s,{x:610,y:330,w:540,h:170,fill:C.light,line:C.line,radius:'rounded-xl'});
 box(s,{x:640,y:360,w:480,h:30,text:'后续要证明的问题',fs:20,color:C.navy,bold:true});
 box(s,{x:640,y:405,w:460,h:86,text:'真实 GPU 上是否保持收益？\n同模型做 StaticCache 与 vLLM A/B。\n能否全部归因于分页？\n不能：引擎还存在调度和算子差异。',fs:16,color:C.gray});
}
// 8 thanks
{
 const s=pres.slides.add();s.background.fill=C.white;
 box(s,{x:0,y:0,w:W,h:18,fill:C.red,line:C.red});
 box(s,{x:0,y:18,w:W,h:8,fill:C.orange,line:C.orange});
 box(s,{x:0,y:260,w:W,h:120,fill:C.pale,line:C.pale});
 box(s,{x:380,y:285,w:520,h:60,text:'谢谢老师，请指导',fs:40,color:C.red,bold:true,align:'center',valign:'center'});
 box(s,{x:290,y:410,w:700,h:30,text:'KV Cache 分页内存管理子系统｜CPU 阶段汇报',fs:18,color:C.gray,align:'center'});
}

const candidate=path.join(outDir,'KV_Cache分页内存管理_正式CPU阶段汇报_50pct.pptx');
const pptx=await PresentationFile.exportPptx(pres); await pptx.save(candidate);
for (const [i,sl] of pres.slides.items.entries()) {
  const png=await pres.export({slide:sl,format:'png',scale:1});
  await fs.writeFile(path.join(buildDir,`updated-slide-${String(i+1).padStart(2,'0')}.png`),new Uint8Array(await png.arrayBuffer()));
}
console.log(candidate);
