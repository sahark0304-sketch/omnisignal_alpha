import os
import sys
import asyncio
from datetime import datetime

# מוודא שפייתון רואה את תיקיית הפרויקט
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from main import process_signal

class MockSignal:
    def __init__(self, text, source):
        self.text = text
        self.content = text
        self.source = source
        self.received_at = datetime.now()
        self.image = None
        self.image_bytes = None
        self.metadata = {}

async def run_test():
    print("============================================================")
    print("  OmniSignal Alpha -- Double Signal (Consensus Fix)")
    print("============================================================")
    
    # איתות ראשון ממקור א'
    # שמתי SL ו-TP הגיוניים לזהב (מתחת ומעל המחיר הנוכחי)
    sig1 = MockSignal(
        text="XAUUSD BUY NOW SL 4600 TP 5800 #TEST_A", 
        source='manual:1'
    )
    
    # איתות שני ממקור ב' (אותו צמד, אותה פעולה) - זה יגרום לבוט לבצע!
    sig2 = MockSignal(
        text="XAUUSD BUY NOW SL 4600 TP 5800 #TEST_B", 
        source='manual:2'
    )
    
    try:
        print("[1/2] Sending first source (Consensus: 1/2)...")
        await process_signal(sig1)
        
        print("[2/2] Sending second source (Consensus: 2/2) -> TRIGGER!")
        await process_signal(sig2)
        
        print("\n✅ SUCCESS: Consensus reached! Order sent to MT5.")
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        
    print("============================================================")

if __name__ == "__main__":
    asyncio.run(run_test())