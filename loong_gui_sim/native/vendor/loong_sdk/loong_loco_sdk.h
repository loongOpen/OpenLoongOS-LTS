/*
Copyright 2025 国家地方共建人形机器人创新中心/人形机器人（上海）有限公司, https://www.openloong.net
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


=====================================================*/
#pragma once
#include<vector>

namespace LocoSdk{
class locoSdkClass{
public:
	static locoSdkClass& instance();
	~locoSdkClass();
	const char* getSdkVersion();
	bool isKeyUsed(const int &planKey);
	const std::vector<int> &getUsedKeys();
	//==强切：用于在plan中手动强制切换plan，谨慎使用。成功返回1，失败返回0
	bool enforceSwitchPlan(const int& tgtPlanKey, const char* switchReason);
	bool enforceSwitchPlan(const char* tgtPlanName, const char* switchReason);
private:
	locoSdkClass();
	locoSdkClass(const locoSdkClass&)=delete;
	locoSdkClass & operator=(const locoSdkClass&)=delete;
	class impClass;
	impClass &imp;
};
}//namespace
