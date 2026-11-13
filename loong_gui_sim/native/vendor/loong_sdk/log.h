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
封装quill，当log到达一定行数，支持自动log新文件
#include"log.h"
=====================================================*/
#pragma once
#include<string>
#include<mutex>
#include"eigen.h"

using namespace std;

namespace Log{
class logClass{
public:
	logClass();
	~logClass();
	void init(string_view suffix);

	logClass& operator<<(const char* data);
	template<typename T>
	logClass& operator<<(const T& data);

	logClass& append(const char* data);
	template<typename T>
	logClass& append(const T& data);

	logClass& appendTab(const char* data);// 每一次会自动追加tab
	template<typename T>
	logClass& appendTab(const T& data);// 每一次会自动追加tab

	logClass& appendComma(const char* data);// 每一次会自动追加逗号
	template<typename T>
	logClass& appendComma(const T& data);// 每一次会自动追加逗号
	
	void log();//将append内容输出
	void log(const char*s);//直接输出
	// void log(const string &s);//直接输出
	void log(string_view s);//直接输出
private:
	class impClass;
	impClass&imp;
};

// ==打印到终端并log到文件，单例========
// append到缓存，log输出
class consoleLogClass{
public:
	~consoleLogClass();
	static consoleLogClass& instance();
	bool openUdp(int port);

	consoleLogClass& operator<<(const char* data);//添加log元素，多线程可能错位，可配合加锁
	template<typename T>
	consoleLogClass& operator<<(const T& data);//添加log元素，多线程可能错位，可配合加锁，数字自动追加逗号

	consoleLogClass& append(const char* data);//添加log元素，多线程可能错位，可配合加锁
	template<typename T>
	consoleLogClass& append(const T& data);//添加log元素，多线程可能错位，可配合加锁，数字自动追加逗号
	template<typename T,int n>
	consoleLogClass& append(const T(&data)[n]);//添加log元素，多线程可能错位，可配合加锁，数组逗号分割

	consoleLogClass& appendTab(const char* data);//添加log元素后自动追加tab，多线程可能错位，可配合加锁
	template<typename T>
	consoleLogClass& appendTab(const T& data);//添加log元素后自动追加tab，多线程可能错位，可配合加锁
	
	consoleLogClass& appendComma(const char* data);//添加log元素后自动追加逗号，多线程可能错位，可配合加锁
	template<typename T>
	consoleLogClass& appendComma(const T& data);//添加log元素后自动追加逗号，多线程可能错位，可配合加锁

	void log();//将append内容输出
	void log(const char*c);//直接输出,字符打印单独写，避免自动数组化模板推导
	// void log(const string &s);//直接输出
	void log(const string& s);//直接输出

	template<typename T,int n>
	void log(const T(&a)[n]);//线程安全，数组,直接输出
	template<typename... T>
	void log(const T&... data);//线程安全,直接输出，两个数字会自动追加逗号分割
	template<typename Derived>
	void logEi(const Eigen::MatrixBase<Derived> &data);//多线程可能错位，可配合加锁，eigen元素逗号分割
	template<typename Derived>
	void logEi(const char*tag, const Eigen::MatrixBase<Derived> &data);//多线程可能错位，可配合加锁，eigen元素逗号分割
private:
	consoleLogClass();
	consoleLogClass(const consoleLogClass&)=delete;
	consoleLogClass & operator=(const consoleLogClass&)=delete;
	class impClass;
	impClass&imp;
	mutex mut;

	template<typename T1, typename... T>
	void expandAppend(const T1& a, const T&... b);
	template<typename T>
	void expandAppend(const T& a);
};
}//namespace

#include"log.hxx"

extern Log::consoleLogClass& conlog;