/*
Copyright 2025 人形机器人（上海）有限公司, https://www.openloong.net
Thanks for the open biped control project Nabo: https://github.com/tryingfly/nabo

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

============ ***doc description @ yyp*** ============
consoleLogClass变参数函数，需在头文件展开
=====================================================*/
#pragma once
#include"log.h"

namespace Log{
	template<typename T,int n> consoleLogClass& consoleLogClass::append(const T(&data)[n]){//多线程可能错位，可配合加锁
		for(int i=0;i<n;i++){
			appendComma(data[i]);
		}
		return *this;
	}
	template<typename Derived> void consoleLogClass::logEi(const Eigen::MatrixBase<Derived> &m){
		auto r=m.rows(), c=m.cols();
		//列向量行打印
		if(c==1){
			For(r){
				if(abs(m(i,0))<1e-4){append("0, ");}
				else{appendComma(m(i,0));}
			}
		}else{
			append('\n');
			For(r){
				for(int j=0;j!=c;++j){
					if(abs(m(i,j))<1e-4){append("0, ");}
					// else{out.width(width);out<<fixed<<setprecision(3)<<m(i,j);}
					else{appendComma(m(i,0));}
				}
				append('\n');
			}
		}
		log();
	}
	template<typename Derived> void consoleLogClass::logEi(const char*tag, const Eigen::MatrixBase<Derived> &m){
		append(tag);append(":\n");
		logEi(m);
	}
	template<typename... T> void consoleLogClass::log(const T&... data){//线程安全,直接输出
		mut.lock();
		expandAppend(data...);
		mut.unlock();
	};
	template<typename T,int n> void consoleLogClass::log(const T(&a)[n]){//线程安全，数组,直接输出
		mut.lock();
		for(int i=0;i<n-1;i++){
			if(abs(a[i])>1e-4){appendComma(a[i]);}
			else{append("0, ");}
		}
		if(abs(a[n-1])>1e-4){append(a[n-1]);}
		else{append('0');}
		log(); 
		mut.unlock();
	}

	template<typename T1, typename... T>
	void consoleLogClass::expandAppend(const T1& a, const T&... b){
		append(a);
		expandAppend(b...);
	}
	template<typename T>
	void consoleLogClass::expandAppend(const T& a){
		append(a);
		log();
	}
	
}//namespace

