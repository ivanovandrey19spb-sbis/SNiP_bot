import os
import time
from supabase import create_client
from dotenv import load_dotenv

load_dotenv(override=True)

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_ANON_KEY = os.environ["SUPABASE_PUBLIC_KEY_LONG"]
EMAIL = os.environ["SUPABASE_EMAIL"]
PASSWORD = os.environ["SUPABASE_PASSWORD"]

print("url ok:", SUPABASE_URL.startswith("https://") and "supabase.co" in SUPABASE_URL)
print("key starts eyJ:", SUPABASE_ANON_KEY.startswith("eyJ"))
print("key dot parts:", len(SUPABASE_ANON_KEY.split(".")))
print("key length:", len(SUPABASE_ANON_KEY))

supabase = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

# 1) Логин (получаем JWT; дальше запросы идут как authenticated)
auth = supabase.auth.sign_in_with_password({"email": EMAIL, "password": PASSWORD})
user = auth.user
print("auth user id:", getattr(user, "id", None))


res = supabase.table("query_cache").select("*").eq("query_hash", "abc123").execute()
print(res.data)






