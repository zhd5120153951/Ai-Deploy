import{C as F,w as b}from"./element-plus.ef27c94c.js";import{_ as v}from"./index.796a76b8.js";import{_ as A}from"./picker.34f033cd.js";import{_ as B}from"./picker.c2491a62.js";import{f as _,b as y}from"./index.b60a26c4.js";import{d as C,o as u,c as r,a as e,W as D,a8 as E,O as U,P as m,X as t,U as w}from"./@vue.a137a740.js";const N={class:"bg-fill-light flex items-center w-full p-4 mb-4"},$={class:"upload-btn w-[60px] h-[60px]"},z={class:"ml-3 flex-1"},P={class:"flex"},I=e("span",{class:"text-tx-regular flex-none mr-3"},"\u540D\u79F0",-1),L={class:"flex mt-[18px]"},O=e("span",{class:"text-tx-regular flex-none mr-3"},"\u94FE\u63A5",-1),H=C({__name:"add-nav",props:{modelValue:{type:Array,default:()=>[]},max:{type:Number,default:10},min:{type:Number,default:1}},setup(d){const l=d,p=()=>{var a;((a=l.modelValue)==null?void 0:a.length)<l.max?l.modelValue.push({image:"",name:"\u5BFC\u822A\u540D\u79F0",link:{}}):_.msgError(`\u6700\u591A\u6DFB\u52A0${l.max}\u4E2A`)},i=a=>{var s;if(((s=l.modelValue)==null?void 0:s.length)<=l.min)return _.msgError(`\u6700\u5C11\u4FDD\u7559${l.min}\u4E2A`);l.modelValue.splice(a,1)};return(a,s)=>{const f=y,x=B,V=F,h=A,g=v,k=b;return u(),r("div",null,[e("div",null,[(u(!0),r(D,null,E(d.modelValue,(o,c)=>(u(),U(g,{class:"max-w-[400px]",key:c,onClose:n=>i(c)},{default:m(()=>[e("div",N,[t(x,{modelValue:o.image,"onUpdate:modelValue":n=>o.image=n,"upload-class":"bg-body",size:"60px"},{upload:m(()=>[e("div",$,[t(f,{name:"el-icon-Plus",size:20})])]),_:2},1032,["modelValue","onUpdate:modelValue"]),e("div",z,[e("div",P,[I,t(V,{modelValue:o.name,"onUpdate:modelValue":n=>o.name=n,placeholder:"\u8BF7\u8F93\u5165\u540D\u79F0"},null,8,["modelValue","onUpdate:modelValue"])]),e("div",L,[O,t(h,{modelValue:o.link,"onUpdate:modelValue":n=>o.link=n},null,8,["modelValue","onUpdate:modelValue"])])])])]),_:2},1032,["onClose"]))),128))]),e("div",null,[t(k,{type:"primary",onClick:p},{default:m(()=>[w("\u6DFB\u52A0")]),_:1})])])}}});export{H as _};
