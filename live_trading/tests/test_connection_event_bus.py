import threading
from live_trading.notifications.connection_event_bus import ConnectionEventBus
from live_trading.notifications.connection_events import ConnectionDropped, AuthFailed


class TestConnectionEventBus:
    def test_subscriber_receives_emitted_event(self):
        bus = ConnectionEventBus()
        received = []
        bus.subscribe(received.append)
        event = ConnectionDropped(reason="test")
        bus.emit(event)
        assert received == [event]

    def test_multiple_subscribers_all_receive_event(self):
        bus = ConnectionEventBus()
        received_a, received_b = [], []
        bus.subscribe(received_a.append)
        bus.subscribe(received_b.append)
        event = AuthFailed(reason="bad token")
        bus.emit(event)
        assert received_a == [event]
        assert received_b == [event]

    def test_no_subscribers_emit_is_noop(self):
        bus = ConnectionEventBus()
        bus.emit(ConnectionDropped(reason="test"))  # must not raise

    def test_subscriber_exception_does_not_propagate(self):
        bus = ConnectionEventBus()
        def bad_subscriber(event):
            raise RuntimeError("subscriber error")
        bus.subscribe(bad_subscriber)
        # Must not raise
        bus.emit(ConnectionDropped(reason="test"))

    def test_emit_is_thread_safe(self):
        """Concurrent emits from multiple threads must not lose events."""
        bus = ConnectionEventBus()
        received = []
        lock = threading.Lock()
        def safe_append(event):
            with lock:
                received.append(event)
        bus.subscribe(safe_append)

        threads = [
            threading.Thread(target=bus.emit, args=(ConnectionDropped(reason=f"t{i}"),))
            for i in range(50)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(received) == 50
