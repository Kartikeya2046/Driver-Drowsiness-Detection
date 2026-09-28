"""Serial link to the Arduino alert loop (context.md: ASCII "R:<level>\\n" at ~2 Hz,
button press -> "ACK\\n" back). Hardware isn't ordered yet (see PROGRESS.md), so this
degrades to logging what it would have sent instead of raising - the software side
must be fully runnable and testable without the physical loop attached.
"""
import time

try:
    import serial
except ImportError:  # pyserial not installed in this env
    serial = None


class SerialClient:
    def __init__(self, port: str | None, baud: int = 9600, timeout: float = 0.05):
        self._ser = None
        if port and serial is not None:
            try:
                self._ser = serial.Serial(port, baud, timeout=timeout)
            except Exception as e:
                print(f"[serial] could not open {port} ({e}); logging only", flush=True)
        elif port:
            print("[serial] pyserial not installed; logging only", flush=True)

    def send_level(self, level: int) -> None:
        msg = f"R:{level}\n"
        if self._ser is not None:
            self._ser.write(msg.encode("ascii"))
        else:
            print(f"[serial] {msg.strip()}", flush=True)

    def poll_ack(self) -> bool:
        """Non-blocking check for a pending ACK\\n from a button press."""
        if self._ser is None or self._ser.in_waiting == 0:
            return False
        line = self._ser.readline().decode("ascii", errors="ignore").strip()
        return line == "ACK"

    def close(self):
        if self._ser is not None:
            self._ser.close()


def demo() -> None:
    c = SerialClient(port=None)  # no hardware -> log-only mode, must not raise
    c.send_level(2)
    assert c.poll_ack() is False
    c.close()
    print("serial_client.py: ok")


if __name__ == "__main__":
    demo()
