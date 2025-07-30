import os
import json
from datetime import datetime
from typing import Dict, List, Optional
import requests
from langchain.chains import LLMChain
from langchain_openai import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate


# --------------------------
# Agent 核心类
# --------------------------
class MultiFunctionAgent:
    def __init__(self):
        # 初始化LLM (建议设置环境变量 OPENAI_API_KEY)
        self.llm = OpenAI(temperature=0.7, model_name="gpt-3.5-turbo",
                          api_key="这里改成自己的apikey")

        # 对话记忆系统
        self.memory = ConversationBufferMemory(memory_key="chat_history")

        # 可用工具集
        self.tools = {
            "calculator": self.calculate,
            "time": self.get_time,
            "weather": self.get_weather,
            "web_search": self.web_search
        }

        # 决策提示模板
        self.decision_prompt = PromptTemplate(
            input_variables=["input", "chat_history", "tools"],
            template="""
            你是一个AI助手，可以访问以下工具：{tools}

            历史对话：
            {chat_history}

            当前请求：{input}

            请选择以下操作之一：
            1. 直接回答（如果不需要工具）
            2. 调用工具（格式：{{"tool": "工具名", "input": "参数"}}）
            3. 要求澄清（如果请求不明确）

            你的响应：
            """
        )

    # --------------------------
    # 工具函数
    # --------------------------
    def calculate(self, expression: str) -> str:
        """数学计算工具"""
        try:
            result = eval(expression)
            return f"计算结果: {expression} = {result}"
        except Exception as e:
            return f"计算错误: {str(e)}"

    def get_time(self, _: Optional[str] = None) -> str:
        """获取当前时间"""
        return f"当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

    def get_weather(self, location: str) -> str:
        """模拟天气查询"""
        # 实际应用中可接入真实API
        weather_data = {
            "北京": "晴，25°C",
            "上海": "多云，28°C",
            "广州": "雷阵雨，30°C"
        }
        return weather_data.get(location, f"未找到{location}的天气信息")

    def web_search(self, query: str) -> str:
        """模拟网络搜索"""
        # 实际应用中可接入SerpAPI等
        return f"网络搜索结果（模拟）: 关于'{query}'的最新信息..."

    # --------------------------
    # 核心处理流程
    # --------------------------
    def process_input(self, user_input: str) -> str:
        # 步骤1：决策
        decision_chain = LLMChain(
            llm=self.llm,
            prompt=self.decision_prompt,
            memory=self.memory
        )

        tool_descriptions = ", ".join([f"{name}: {func.__doc__}" for name, func in self.tools.items()])

        response = decision_chain.run(
            input=user_input,
            tools=tool_descriptions
        )

        # 步骤2：解析决策
        try:
            # 尝试解析JSON工具调用
            tool_call = json.loads(response)
            if "tool" in tool_call:
                tool_name = tool_call["tool"]
                tool_input = tool_call.get("input", "")

                if tool_name in self.tools:
                    # 步骤3：执行工具
                    tool_result = self.tools[tool_name](tool_input)
                    self.memory.save_context(
                        {"input": user_input},
                        {"output": f"调用工具[{tool_name}] 结果: {tool_result}"}
                    )
                    return tool_result
        except json.JSONDecodeError:
            pass  # 不是工具调用

        # 步骤4：直接返回文本响应
        self.memory.save_context(
            {"input": user_input},
            {"output": response}
        )
        return response


# --------------------------
# 测试运行
# --------------------------
if __name__ == "__main__":
    agent = MultiFunctionAgent()

    test_cases = [
        "北京现在几点？",
        "计算3的平方加4的平方",
        "上海天气怎么样？",
        "特斯拉最新新闻",
        "请告诉我关于宇宙的趣事"
    ]

    for query in test_cases:
        print(f"用户: {query}")
        response = agent.process_input(query)
        print(f"Agent: {response}\n{'=' * 50}")