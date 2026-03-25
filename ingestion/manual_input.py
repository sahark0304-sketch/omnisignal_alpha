"""
ingestion/manual_input.py — WhatsApp/paste entry point.

Provides a simple interface to manually submit signals that were
received via WhatsApp, email, or any non-automated source.
Can be used as a CLI tool or imported for programmatic use.
"""

import asyncio
from datetime import datetime
from ingestion.signal_queue import push, RawSignal
from utils.logger import get_logger

logger = get_logger(__name__)


async def submit_signal(text: str, source: str = "manual"):
    """Submit a manually entered signal to the processing queue."""
    if not text or len(text.strip()) < 5:
        logger.warning("[Manual] Signal text too short — ignoring.")
        return False

    raw = RawSignal(
        content=text.strip(),
        source=f"manual:{source}",
        received_at=datetime.now(),
    )
    await push(raw)
    logger.info(f"[Manual] Signal submitted: {text[:60]}...")
    return True


async def interactive_mode():
    """Interactive CLI for pasting signals."""
    print("\n" + "=" * 50)
    print("  OmniSignal Alpha — Manual Signal Input")
    print("  Type a signal and press Enter. Type 'quit' to exit.")
    print("=" * 50 + "\n")

    while True:
        try:
            text = input("📝 Paste signal > ").strip()
            if text.lower() in ("quit", "exit", "q"):
                break
            if text:
                source = input("   Source name (default: whatsapp) > ").strip() or "whatsapp"
                success = await submit_signal(text, source)
                if success:
                    print("   ✅ Signal queued for processing.\n")
                else:
                    print("   ❌ Signal too short.\n")
        except (KeyboardInterrupt, EOFError):
            break

    print("\n👋 Manual input closed.")


if __name__ == "__main__":
    asyncio.run(interactive_mode())
