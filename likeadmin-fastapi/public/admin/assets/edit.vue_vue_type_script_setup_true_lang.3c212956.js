import{C as k,D as x,v as R,t as h,F as y}from"./element-plus.ef27c94c.js";import{p as S,a as A,b as U}from"./post.2ca46716.js";import{P as I}from"./index.d37c8696.js";import{f as N}from"./index.b60a26c4.js";import{d as P,s as f,r as q,e as z,a1 as T,o as X,c as j,X as l,P as s,u as t,a as F}from"./@vue.a137a740.js";const G={class:"edit-popup"},H=F("div",{class:"form-tips"},"\u9ED8\u8BA4\u4E3A0\uFF0C \u6570\u503C\u8D8A\u5927\u8D8A\u6392\u524D",-1),W=P({__name:"edit",emits:["success","close"],setup(J,{expose:_,emit:p}){const i=f(),n=f(),m=q("add"),D=z(()=>m.value=="edit"?"\u7F16\u8F91\u5C97\u4F4D":"\u65B0\u589E\u5C97\u4F4D"),u=T({id:"",name:"",code:"",sort:0,remarks:"",isStop:0}),C={code:[{required:!0,message:"\u8BF7\u8F93\u5165\u5C97\u4F4D\u7F16\u7801",trigger:["blur"]}],name:[{required:!0,message:"\u8BF7\u8F93\u5165\u5C97\u4F4D\u540D\u79F0",trigger:["blur"]}]},v=async()=>{var o,e;await((o=i.value)==null?void 0:o.validate()),m.value=="edit"?await S(u):await A(u),N.msgSuccess("\u64CD\u4F5C\u6210\u529F"),(e=n.value)==null||e.close(),p("success")},V=(o="add")=>{var e;m.value=o,(e=n.value)==null||e.open()},c=o=>{for(const e in u)o[e]!=null&&o[e]!=null?u[e]=o[e]:u[e]=o.is_stop},b=async o=>{const e=await U({id:o.id});c(e)},w=()=>{p("close")};return _({open:V,setFormData:c,getDetail:b}),(o,e)=>{const d=k,r=x,E=R,B=h,g=y;return X(),j("div",G,[l(I,{ref_key:"popupRef",ref:n,title:t(D),async:!0,width:"550px",onConfirm:v,onClose:w},{default:s(()=>[l(g,{ref_key:"formRef",ref:i,model:t(u),"label-width":"84px",rules:C},{default:s(()=>[l(r,{label:"\u5C97\u4F4D\u540D\u79F0",prop:"name"},{default:s(()=>[l(d,{modelValue:t(u).name,"onUpdate:modelValue":e[0]||(e[0]=a=>t(u).name=a),placeholder:"\u8BF7\u8F93\u5165\u5C97\u4F4D\u540D\u79F0",clearable:"",maxlength:100},null,8,["modelValue"])]),_:1}),l(r,{label:"\u5C97\u4F4D\u7F16\u7801",prop:"code"},{default:s(()=>[l(d,{modelValue:t(u).code,"onUpdate:modelValue":e[1]||(e[1]=a=>t(u).code=a),placeholder:"\u8BF7\u8F93\u5165\u5C97\u4F4D\u7F16\u7801",clearable:""},null,8,["modelValue"])]),_:1}),l(r,{label:"\u6392\u5E8F",prop:"sort"},{default:s(()=>[F("div",null,[l(E,{modelValue:t(u).sort,"onUpdate:modelValue":e[2]||(e[2]=a=>t(u).sort=a),min:0,max:9999},null,8,["modelValue"]),H])]),_:1}),l(r,{label:"\u5907\u6CE8",prop:"remarks"},{default:s(()=>[l(d,{modelValue:t(u).remarks,"onUpdate:modelValue":e[3]||(e[3]=a=>t(u).remarks=a),placeholder:"\u8BF7\u8F93\u5165\u5907\u6CE8",type:"textarea",autosize:{minRows:4,maxRows:6},maxlength:"200","show-word-limit":""},null,8,["modelValue"])]),_:1}),l(r,{label:"\u5C97\u4F4D\u72B6\u6001",prop:"isStop"},{default:s(()=>[l(B,{modelValue:t(u).isStop,"onUpdate:modelValue":e[4]||(e[4]=a=>t(u).isStop=a),"active-value":0,"inactive-value":1},null,8,["modelValue"])]),_:1})]),_:1},8,["model"])]),_:1},8,["title"])])}}});export{W as _};
