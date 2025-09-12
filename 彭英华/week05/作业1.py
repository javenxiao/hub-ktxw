import openai
client = openai.OpenAI(base_url="https://api.deepseek.com",api_key=" sk-f8d7c521c701422487b51a826c7bc726")
completion = client.chat.completions.create(model="deepseek-chat",
                                            messages=[{"role":"system","content":"you are a helpful assistant"},
                                                      {"role":"user","content":"你是谁"}],
                                            stream=False)
print(completion.choices[0].message.content)
