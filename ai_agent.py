import os
import json
import re
import math

from dotenv import load_dotenv
from openai import OpenAI

# 加载环境变量（存储 API 密钥）
load_dotenv()
# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
client = OpenAI(api_key="这里改成自己的apikey")



class AIAgent:
    def __init__(self, name: str):
        self.name = name
        self.memory = []  # 存储对话历史（短期记忆）
        self.tools = {  # 注册可用工具
            "calculator": self.calculate,
            "查天气": self.get_weather
        }

    def add_memory(self, role: str, content: str):
        """添加对话到记忆"""
        self.memory.append({"role": role, "content": content})

    def calculate(self, expression: str) -> str:
        """计算器工具：执行数学运算"""
        try:
            # # 替换常见的数学符号为Python支持的格式
            # expression = re.sub(r'x|\*', '*', expression)  # 统一乘法符号
            # expression = re.sub(r'÷|/', '/', expression)  # 统一除法符号
            # expression = re.sub(r'\^', '**', expression)  # 替换幂运算符号
            #
            # # 限制只能进行数学运算，提高安全性
            # allowed_chars = set('0123456789+-*/(). ')
            # if not all(c in allowed_chars for c in expression):
            #     return "计算失败：表达式包含不允许的字符"

            result = eval(expression)  # 执行计算
            return f"计算结果：{expression} = {result}"
        except Exception as e:
            return f"计算失败：{str(e)}"

    def get_weather(self, city: str) -> str:
        """天气查询工具（模拟）"""
        # 实际应用中可调用天气 API
        mock_weather = {
            "北京": "晴，25°C",
            "上海": "多云，28°C"
        }
        return f"{city} 的天气：{mock_weather.get(city, '未知')}"

    def decide_action(self, user_input: str) -> dict:
        """决策模块：判断是否需要调用工具或直接回答"""
        # 提示词：告诉模型如何决策和调用工具
        prompt = """
        你是一个 AI 助手，你的名字是超级厉害的人工智能，可以调用工具解决问题。
        - 其他情况直接回答。
        - 如果需要计算，必须先把用户输入转换为python的数学计算公式，对数计算必须使用math.log(数值, 底数)，例如log₃(81)应写为math.log(81, 3)，必须使用工具：{"action": "calculator", "params": {"expression": "数学表达式"}}

        - 如果需要查询天气，使用工具：{"action": "查天气", "params": {"city": "城市名称"}}

        """
        self.add_memory("user", user_input)

        # 调用大语言模型生成决策
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": prompt},
                *self.memory  # 传入历史对话
            ]
        )

        result = response.choices[0].message.content
        self.add_memory("assistant", result)

        # 解析是否需要调用工具
        try:
            action = json.loads(result)
            if "action" in action and action["action"] in self.tools:
                return action  # 返回工具调用指令
        except:
            pass
        return {"action": "answer", "content": result}  # 直接回答

    def run(self, user_input: str) -> str:
        """执行流程：感知 → 决策 → 执行"""
        action = self.decide_action(user_input)

        if action["action"] == "answer":
            return action["content"]
        else:
            # 调用工具并返回结果
            tool = self.tools[action["action"]]
            result = tool(**action["params"])
            self.add_memory("system", f"工具返回：{result}")
            return result


# 使用示例
if __name__ == "__main__":
    agent = AIAgent("智能助手")

    # 测试直接回答
    print(agent.run("你叫什么名字？"))  # 输出：我是智能助手

    # 测试调用计算器
    print(agent.run("使用工具计算81以3底的对数是多少"))  # 输出：计算结果：3+5*2 = 13

    # 测试调用天气工具
    print(agent.run("北京的天气怎么样？"))  # 输出：北京 的天气：晴，25°C
