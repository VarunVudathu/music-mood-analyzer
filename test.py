import re

test = "hello - aiwd"

test = re.sub("(?<=\s\-\s)(.*)", "", test)
test = test.replace(" - ", "")
print(test)


# print(test)
# print("---------")
# test = re.sub("(?<=[a-z])(?=['][A-Z0-9])", " ", test)
# test = re.sub("(?<=[a-z])(?=[A-Z0-9])", " ", test)
# print(test)
# print("---------")
# test = test.lower()
# print(test)
# print("---------")
# test = test.strip().replace("in'", "ing").replace("i'ma", "i am going to").replace("'em", "them").replace("'cause", "because").replace(",", "")
# print(test)
# print("---------")
# test = re.sub("\[(.*?)\]", " ", test)
# print(test)
# print("---------")
# test = re.sub("\((.*?)\)", " ", test)
# print(test)
# print("---------")
# test = test.strip()
# print(test)
# print("---------")

# def access_code():
#     client_creds = f"{os.getenv('CLIENT_ID')}:{os.getenv('CLIENT_SECRET')}"
#     client_creds_b64 = base64.b64encode(client_creds.encode())

#     token_url = "https://accounts.spotify.com/api/token"
#     method = "POST"
#     token_data = {
#         "grant_type": "client_credentials"
#     }
#     token_headers = {
#         "Authorization": f"Basic {client_creds_b64.decode()}"
#     }

#     r = requests.post(token_url, data=token_data, headers=token_headers)
#     print(r.json())