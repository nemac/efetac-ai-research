from distributed import Client
client = Client('152.18.148.195:8786')

def test():
    return 'success!'

result = client.submit(test)
