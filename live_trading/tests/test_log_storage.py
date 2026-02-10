"""
Tests for the log storage and filtering functionality.
"""
import pytest
import json
import tempfile
import shutil
import os
from datetime import datetime, timedelta
from pathlib import Path

from live_trading.logging.log_storage import LogEntry, FileLogStorage


class TestLogEntry:
    """Tests for the LogEntry dataclass"""

    def test_create_log_entry(self):
        entry = LogEntry(
            timestamp="2026-02-06T12:00:00Z",
            level="INFO",
            logger="live_trading.engine",
            message="Operation started"
        )
        assert entry.timestamp == "2026-02-06T12:00:00Z"
        assert entry.level == "INFO"
        assert entry.logger == "live_trading.engine"
        assert entry.message == "Operation started"
        assert entry.extra is None

    def test_create_log_entry_with_extra(self):
        entry = LogEntry(
            timestamp="2026-02-06T12:00:00Z",
            level="ERROR",
            logger="live_trading.brokers",
            message="Connection failed",
            extra={"operation_id": "abc123", "broker": "CTRADER"}
        )
        assert entry.extra == {"operation_id": "abc123", "broker": "CTRADER"}

    def test_to_dict(self):
        entry = LogEntry(
            timestamp="2026-02-06T12:00:00Z",
            level="WARNING",
            logger="test",
            message="test message",
            extra={"key": "value"}
        )
        d = entry.to_dict()
        assert d == {
            "timestamp": "2026-02-06T12:00:00Z",
            "level": "WARNING",
            "logger": "test",
            "message": "test message",
            "extra": {"key": "value"}
        }

    def test_to_dict_without_extra(self):
        entry = LogEntry(
            timestamp="2026-02-06T12:00:00Z",
            level="INFO",
            logger="test",
            message="test message"
        )
        d = entry.to_dict()
        assert d["extra"] is None

    def test_from_dict(self):
        data = {
            "timestamp": "2026-02-06T12:00:00Z",
            "level": "INFO",
            "logger": "test",
            "message": "hello"
        }
        entry = LogEntry.from_dict(data)
        assert entry.level == "INFO"
        assert entry.message == "hello"

    def test_json_roundtrip(self):
        original = LogEntry(
            timestamp="2026-02-06T12:00:00Z",
            level="ERROR",
            logger="test.module",
            message="Something broke",
            extra={"stack": "line 42"}
        )
        json_line = original.to_json_line()
        restored = LogEntry.from_json_line(json_line)
        assert original.timestamp == restored.timestamp
        assert original.level == restored.level
        assert original.logger == restored.logger
        assert original.message == restored.message
        assert original.extra == restored.extra


class TestFileLogStorage:
    """Tests for file-based log storage"""

    @pytest.fixture
    def temp_log_dir(self):
        """Create a temporary directory for log tests"""
        d = tempfile.mkdtemp()
        yield d
        shutil.rmtree(d)

    @pytest.fixture
    def storage(self, temp_log_dir):
        """Create a FileLogStorage instance with temp directory"""
        return FileLogStorage(
            log_dir=temp_log_dir,
            max_file_size_mb=1,
            max_files=5
        )

    def _make_entry(self, level="INFO", logger="test", message="test msg",
                    timestamp=None, extra=None):
        """Helper to create a LogEntry"""
        if timestamp is None:
            timestamp = datetime.utcnow().isoformat() + "Z"
        return LogEntry(
            timestamp=timestamp,
            level=level,
            logger=logger,
            message=message,
            extra=extra
        )

    def test_write_and_read(self, storage):
        """Test basic write and read"""
        entry = self._make_entry(message="hello world")
        storage.write(entry)

        results = storage.read()
        assert len(results) == 1
        assert results[0].message == "hello world"

    def test_write_multiple_and_read(self, storage):
        """Test writing multiple entries"""
        for i in range(5):
            storage.write(self._make_entry(message=f"message {i}"))

        results = storage.read()
        assert len(results) == 5

    def test_read_returns_newest_first(self, storage):
        """Test that read returns entries in newest-first order"""
        storage.write(self._make_entry(
            message="first",
            timestamp="2026-02-06T10:00:00Z"
        ))
        storage.write(self._make_entry(
            message="second",
            timestamp="2026-02-06T11:00:00Z"
        ))
        storage.write(self._make_entry(
            message="third",
            timestamp="2026-02-06T12:00:00Z"
        ))

        results = storage.read()
        assert len(results) == 3
        # Newest first
        assert results[0].message == "third"
        assert results[1].message == "second"
        assert results[2].message == "first"

    def test_filter_by_level(self, storage):
        """Test filtering by log level"""
        storage.write(self._make_entry(level="INFO", message="info msg"))
        storage.write(self._make_entry(level="ERROR", message="error msg"))
        storage.write(self._make_entry(level="WARNING", message="warn msg"))
        storage.write(self._make_entry(level="INFO", message="info msg 2"))

        results = storage.read(level="ERROR")
        assert len(results) == 1
        assert results[0].message == "error msg"

    def test_filter_by_level_case_insensitive(self, storage):
        """Test that level filtering is case insensitive"""
        storage.write(self._make_entry(level="ERROR", message="error msg"))
        storage.write(self._make_entry(level="INFO", message="info msg"))

        results = storage.read(level="error")
        assert len(results) == 1
        assert results[0].level == "ERROR"

    def test_filter_by_logger(self, storage):
        """Test filtering by logger name (partial match)"""
        storage.write(self._make_entry(logger="live_trading.brokers.ctrader", message="ctrader log"))
        storage.write(self._make_entry(logger="live_trading.engine", message="engine log"))
        storage.write(self._make_entry(logger="live_trading.brokers.ibkr", message="ibkr log"))

        results = storage.read(logger="brokers")
        assert len(results) == 2
        loggers = {r.logger for r in results}
        assert "live_trading.brokers.ctrader" in loggers
        assert "live_trading.brokers.ibkr" in loggers

    def test_filter_by_logger_case_insensitive(self, storage):
        """Test that logger filtering is case insensitive"""
        storage.write(self._make_entry(logger="live_trading.Engine", message="test"))

        results = storage.read(logger="engine")
        assert len(results) == 1

    def test_filter_by_search(self, storage):
        """Test search in message content"""
        storage.write(self._make_entry(message="[CONNECTION] Connected to broker"))
        storage.write(self._make_entry(message="[ORDER] Buy order placed"))
        storage.write(self._make_entry(message="[CONNECTION] Disconnected"))

        results = storage.read(search="CONNECTION")
        assert len(results) == 2

    def test_filter_by_search_case_insensitive(self, storage):
        """Test that search is case insensitive"""
        storage.write(self._make_entry(message="Error connecting to broker"))

        results = storage.read(search="error")
        assert len(results) == 1

    def test_search_in_extra_field(self, storage):
        """Test that search also checks extra field"""
        storage.write(self._make_entry(
            message="Operation failed",
            extra={"operation_id": "abc123"}
        ))
        storage.write(self._make_entry(message="Other message"))

        results = storage.read(search="abc123")
        assert len(results) == 1
        assert results[0].message == "Operation failed"

    def test_combined_filters(self, storage):
        """Test combining multiple filters"""
        storage.write(self._make_entry(level="ERROR", logger="live_trading.brokers", message="[CONNECTION] Failed"))
        storage.write(self._make_entry(level="INFO", logger="live_trading.brokers", message="[CONNECTION] Connected"))
        storage.write(self._make_entry(level="ERROR", logger="live_trading.engine", message="Strategy error"))
        storage.write(self._make_entry(level="ERROR", logger="live_trading.brokers", message="[ORDER] Failed"))

        results = storage.read(level="ERROR", logger="brokers", search="CONNECTION")
        assert len(results) == 1
        assert results[0].message == "[CONNECTION] Failed"

    def test_filter_by_start_time(self, storage):
        """Test filtering by start time"""
        storage.write(self._make_entry(
            message="old",
            timestamp="2026-02-05T10:00:00Z"
        ))
        storage.write(self._make_entry(
            message="new",
            timestamp="2026-02-06T10:00:00Z"
        ))

        start = datetime(2026, 2, 6, 0, 0, 0)
        results = storage.read(start_time=start)
        assert len(results) == 1
        assert results[0].message == "new"

    def test_filter_by_end_time(self, storage):
        """Test filtering by end time"""
        storage.write(self._make_entry(
            message="old",
            timestamp="2026-02-05T10:00:00Z"
        ))
        storage.write(self._make_entry(
            message="new",
            timestamp="2026-02-06T10:00:00Z"
        ))

        end = datetime(2026, 2, 5, 23, 59, 59)
        results = storage.read(end_time=end)
        assert len(results) == 1
        assert results[0].message == "old"

    def test_filter_by_date_range(self, storage):
        """Test filtering by date range"""
        storage.write(self._make_entry(message="day1", timestamp="2026-02-04T10:00:00Z"))
        storage.write(self._make_entry(message="day2", timestamp="2026-02-05T10:00:00Z"))
        storage.write(self._make_entry(message="day3", timestamp="2026-02-06T10:00:00Z"))
        storage.write(self._make_entry(message="day4", timestamp="2026-02-07T10:00:00Z"))

        start = datetime(2026, 2, 5, 0, 0, 0)
        end = datetime(2026, 2, 6, 23, 59, 59)
        results = storage.read(start_time=start, end_time=end)
        assert len(results) == 2
        messages = {r.message for r in results}
        assert "day2" in messages
        assert "day3" in messages

    def test_limit(self, storage):
        """Test limit parameter"""
        for i in range(10):
            storage.write(self._make_entry(message=f"msg {i}"))

        results = storage.read(limit=3)
        assert len(results) == 3

    def test_offset(self, storage):
        """Test offset parameter for pagination"""
        for i in range(10):
            storage.write(self._make_entry(
                message=f"msg {i}",
                timestamp=f"2026-02-06T{10+i:02d}:00:00Z"
            ))

        # First page
        page1 = storage.read(limit=3, offset=0)
        assert len(page1) == 3
        assert page1[0].message == "msg 9"  # newest first

        # Second page
        page2 = storage.read(limit=3, offset=3)
        assert len(page2) == 3
        assert page2[0].message == "msg 6"

    def test_empty_results(self, storage):
        """Test reading from empty storage"""
        results = storage.read()
        assert results == []

    def test_no_matching_filter(self, storage):
        """Test filter that matches nothing"""
        storage.write(self._make_entry(level="INFO", message="hello"))

        results = storage.read(level="CRITICAL")
        assert results == []

    def test_log_rotation(self, temp_log_dir):
        """Test log file rotation when max size is exceeded"""
        # Use tiny file size to trigger rotation
        storage = FileLogStorage(
            log_dir=temp_log_dir,
            max_file_size_mb=0,  # Will be 0 bytes, so first write triggers rotation check
            max_files=3
        )

        # Write enough entries to trigger rotation
        # max_file_size = 0, so every write after the first should rotate
        storage.write(self._make_entry(message="first"))
        storage.write(self._make_entry(message="second"))
        storage.write(self._make_entry(message="third"))

        # Verify rotated files exist
        log_dir = Path(temp_log_dir)
        current = log_dir / "live_trading.log"
        rotated_1 = log_dir / "live_trading.log.1"

        assert current.exists()
        # At least one rotation should have happened
        assert rotated_1.exists() or (log_dir / "live_trading.log.2").exists()

    def test_read_across_rotated_files(self, temp_log_dir):
        """Test that read works across current and rotated files"""
        storage = FileLogStorage(
            log_dir=temp_log_dir,
            max_file_size_mb=0,  # Force rotation
            max_files=5
        )

        # Write entries that will span multiple files
        for i in range(5):
            storage.write(self._make_entry(
                message=f"msg {i}",
                timestamp=f"2026-02-06T{10+i:02d}:00:00Z"
            ))

        results = storage.read()
        assert len(results) >= 1  # At least some entries survive rotation

    def test_get_stats(self, storage):
        """Test statistics reporting"""
        storage.write(self._make_entry(level="INFO", message="info"))
        storage.write(self._make_entry(level="ERROR", message="error"))
        storage.write(self._make_entry(level="INFO", message="info2"))

        stats = storage.get_stats()
        assert "log_directory" in stats
        assert "current_file_size_mb" in stats
        assert "level_counts" in stats
        assert stats["level_counts"]["INFO"] == 2
        assert stats["level_counts"]["ERROR"] == 1

    def test_malformed_json_lines_are_skipped(self, temp_log_dir):
        """Test that malformed JSON lines don't crash the reader"""
        storage = FileLogStorage(log_dir=temp_log_dir)

        # Write a valid entry
        storage.write(self._make_entry(message="valid"))

        # Manually append a malformed line
        log_file = Path(temp_log_dir) / "live_trading.log"
        with open(log_file, 'a') as f:
            f.write("this is not json\n")

        # Write another valid entry
        storage.write(self._make_entry(message="also valid"))

        results = storage.read()
        assert len(results) == 2
        messages = {r.message for r in results}
        assert "valid" in messages
        assert "also valid" in messages

    def test_search_with_special_characters(self, storage):
        """Test search with characters that could break regex"""
        storage.write(self._make_entry(message="Error at [line 42] in (module)"))
        storage.write(self._make_entry(message="Normal message"))

        results = storage.read(search="[line 42]")
        assert len(results) == 1

    def test_concurrent_writes(self, storage):
        """Test that concurrent writes don't corrupt the log file"""
        import threading

        def write_entries(prefix, count):
            for i in range(count):
                storage.write(self._make_entry(message=f"{prefix}-{i}"))

        threads = [
            threading.Thread(target=write_entries, args=("t1", 20)),
            threading.Thread(target=write_entries, args=("t2", 20)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        results = storage.read(limit=100)
        assert len(results) == 40

    def test_cleanup_old_archives(self, temp_log_dir):
        """Test cleanup of old archive files"""
        storage = FileLogStorage(log_dir=temp_log_dir)
        archive_dir = Path(temp_log_dir) / "archive"

        # Create fake archive files with old dates
        old_file = archive_dir / "2025-01-01.log.gz"
        recent_file = archive_dir / "2026-02-05.log.gz"
        old_file.write_text("old")
        recent_file.write_text("recent")

        result = storage.cleanup(older_than_days=30)
        assert result["removed_files"] == 1
        assert not old_file.exists()
        assert recent_file.exists()
