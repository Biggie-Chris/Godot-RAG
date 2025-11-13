import os
from dotenv import load_dotenv
from openai import OpenAI

def load_environment():
    """加载环境变量"""
    load_dotenv()
    
    api_key = os.getenv('OPENAI_API_KEY')
    base_url = os.getenv('OPENAI_BASE_URL')
    
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in .env file")
    if not base_url:
        raise ValueError("OPENAI_BASE_URL not found in .env file")
    
    print(f"✅ 成功加载环境变量")
    print(f"   API Base URL: {base_url}")
    print(f"   API Key: {api_key[:10]}...{api_key[-4:]}")
    
    return api_key, base_url

def initialize_openai_client(api_key, base_url):
    """初始化OpenAI客户端"""
    client = OpenAI(
        api_key=api_key,
        base_url=base_url
    )
    print("✅ OpenAI客户端初始化成功")
    return client

def chat_with_ai(client, model="Qwen/Qwen3-32B"):
    """与AI进行对话"""
    print(f"\n🤖 开始与 {model} 对话 (输入 'quit' 退出)")
    print("-" * 50)
    
    conversation_history = []
    
    while True:
        try:
            user_input = input("\n👤 你: ").strip()
            
            if user_input.lower() in ['quit', 'exit', '退出']:
                print("👋 再见！")
                break
                
            if not user_input:
                print("⚠️  请输入有效内容")
                continue
            
            # 添加用户消息到对话历史
            conversation_history.append({"role": "user", "content": user_input})
            
            print("🤔 AI正在思考...")
            
            # 调用API
            response = client.chat.completions.create(
                model=model,
                messages=conversation_history,
                max_tokens=500,
                temperature=0.7
            )
            
            # 获取AI回复
            ai_response = response.choices[0].message.content
            print(f"\n🤖 AI: {ai_response}")
            
            # 添加AI回复到对话历史
            conversation_history.append({"role": "assistant", "content": ai_response})
            
            # 显示token使用情况
            usage = response.usage
            print(f"📊 Token使用: 输入{usage.prompt_tokens} / 输出{usage.completion_tokens} / 总计{usage.total_tokens}")
            
        except KeyboardInterrupt:
            print("\n\n👋 用户中断，再见！")
            break
        except Exception as e:
            print(f"❌ 发生错误: {e}")
            print("请检查网络连接和API配置")

def test_api_connection(client, model="Qwen/Qwen3-32B"):
    """测试API连接"""
    print("🔍 测试API连接...")
    
    try:
        # 发送一个简单的测试消息
        test_messages = [{"role": "user", "content": "请简单回复'连接成功'"}]
        
        response = client.chat.completions.create(
            model=model,
            messages=test_messages,
            max_tokens=10
        )
        
        test_response = response.choices[0].message.content
        print(f"✅ API连接测试成功: {test_response}")
        return True
        
    except Exception as e:
        print(f"❌ API连接测试失败: {e}")
        return False

def main():
    """主函数"""
    print("🚀 大模型API对话测试程序")
    print("=" * 50)
    
    try:
        # 1. 加载环境变量
        api_key, base_url = load_environment()
        
        # 2. 初始化客户端
        client = initialize_openai_client(api_key, base_url)
        
        # 3. 测试API连接
        if test_api_connection(client):
            # 4. 开始对话
            chat_with_ai(client)
        else:
            print("❌ API连接失败，请检查配置")
            
    except Exception as e:
        print(f"❌ 程序初始化失败: {e}")
        print("请检查.env文件配置是否正确")

if __name__ == "__main__":
    main()