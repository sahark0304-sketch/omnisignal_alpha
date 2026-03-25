import MetaTrader5 as mt5
import os

# הנתיב המדויק שראינו בתמונה שלך
MT5_PATH=C:/Program Files/MetaTrader 5 IC Markets Global/terminal64.exe

# ניסיון חיבור עם הנתיב המפורש
if not mt5.initialize(path=mt5_path):
    print("❌ Connection Failed!")
    print("Error details:", mt5.last_error())
else:
    print("✅ Connected: True")
    # ניסיון התחברות לחשבון
    login = "52786779"  
    password = "WLbvoyXa$9WBLK"
    server = "ICMarketsSC-Demo"
    
    if mt5.login(login, password, server):
         acc = mt5.account_info()
         print(f"Account: {acc.login} | Balance: ${acc.balance:.2f}")
    else:
         print("❌ Login failed:", mt5.last_error())
    
    mt5.shutdown()