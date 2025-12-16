"""
Utility classes and functions for windIO integration.

This module contains shared utilities used across the windIO integration:
- TrackedDict: Dictionary that tracks field access
- extract_data: Helper for extracting dimensional data
- WAKE_MODEL_MAPPING: Mapping between windIO and FLORIS wake models
"""

from pathlib import Path
import numpy as np
from typing import Dict, Any, List
from collections import UserDict

from floris.logging_manager import LoggingManager


class TrackedDict(UserDict, LoggingManager):
    """
    Dictionary that tracks which keys have been accessed.
    Recursively converts nested dicts to TrackedDict.
    """

    def __init__(self, data: Dict[str, Any], context: str = None, parent: "TrackedDict" = None):
        if isinstance(data, TrackedDict):
            raise TypeError("Cannot initialize TrackedDict with another TrackedDict")
        
        super().__init__()

        self._context = context or data.pop("_context", ".")
        self._tracked_keys = TrackedDict._init_tracked_keys(data)
        self._read_keys = set()  # All keys start unread
        self._nested_dicts = {}
        self._parent_dict = parent
        self._tracking_status = True

        for key, value in data.items():
            if isinstance(value, TrackedDict):
                raise TypeError("Nested TrackedDicts are not allowed during initialization")
            
            elif isinstance(value, dict):
                nested = TrackedDict(value, context=self._context + "." + str(key), parent=self)
                self.data[key] = nested
                self._nested_dicts[key] = nested
                
            else:
                self.data[key] = value
               
    @staticmethod
    def _init_tracked_keys(data: Dict[str, Any]) -> set:
        tk = set()
        for k in data.keys():
            if isinstance(k, str):
                if not k.startswith("_"):
                    tk.add(k)
            else:
                tk.add(k)
        return tk

    @staticmethod
    def from_parent(data: Dict[str, Any], key: str) -> "TrackedDict":
        if isinstance(data, TrackedDict):
            raise TypeError("Cannot create nested TrackedDict from another TrackedDict")

        if isinstance(data, dict):
            return TrackedDict(data[key], context=data.get("_context", "") + "." + key)
        
        raise TypeError(f"Cannot create TrackedDict from type {type(data)}")
    
    @staticmethod
    def from_list(data_list: List[Dict[str, Any]], context: str) -> List["TrackedDict"]:
        data_dict = {i: data for i, data in enumerate(data_list)}
        return TrackedDict(data_dict, context=context)
    
    @property
    def context(self) -> str:
        return self._context

    @property
    def tracked_keys(self) -> List[str]:
        """List of keys that are being tracked."""
        return list(self._tracked_keys)
    
    def mark_read(self, key: str):
        """Manually mark a key as read."""
        if not self._tracking_status:
            return
        
        if key in self._tracked_keys:
            self._read_keys.add(key)
        elif key in self.data:
            # Key exists but is not tracked
            self.logger.warning(f"Key '{self._context}.{key}' exists but is not tracked")
        else:
            raise KeyError(f"Key '{key}' not found in '{self._context}'")

    def untrack(self, key: str):
        """Detach a nested TrackedDict from tracking."""
        if (key in self.tracked_keys) and (key in self._nested_dicts):
            # Buffer data
            nested = self._nested_dicts.pop(key).data

            # Convert to raw dict and remove references
            self.data[key] = nested  # Replace with raw dict
            self._tracked_keys.remove(key)
            self._read_keys.discard(key)
            return nested
        else:
            raise KeyError(f"Key '{key}' not found in nested TrackedDicts of '{self._context}'")

    def __getitem__(self, key: str) -> Any:
        if key not in self.data:
            raise KeyError(f"Key '{key}' not found in '{self._context}'")
        
        self.mark_read(key)
        return self.data[key]
    
    def get(self, key: str, default: Any = None) -> Any:
        if key in self.data:
            self.mark_read(key)
            return self.data[key]
        self.logger.debug(f"Key '{key}' not found in '{self._context}', returning default value ({default})")
        return default

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self, issue_warning: bool = True):
        self._tracking_status = False  # Disable tracking during close

        unread_data = []
        if not self.all_read:
            for key in self.tracked_keys:
                if isinstance(self.data[key], TrackedDict):
                    # print("Found tracked dict at key:", key)
                    # If no nested keys were read, TrackedDict was never accessed, raise warning
                    if (not self.data[key].any_read) and (issue_warning):
                        unread_data.append(key)
                        # Warning only to be issued at this level 
                        self.data[key].close(issue_warning=False)

                    # If some nested keys were read, TrackedDict was accessed, close normally but need to check underlying unread keys
                    else:
                        self.data[key].close(issue_warning=issue_warning)

                elif (issue_warning):
                    unread_data.append(key)

            if unread_data:
                # print(self)
                self.logger.warning(f"Unread keys detected in {self._context}: {unread_data}")

        self._tracking_status = True  # Re-enable tracking

    @property
    def read_keys(self) -> List[str]:
        """List of keys that have been read."""
        return list(self._read_keys)
    
    @property
    def unread_keys(self) -> List[str]:
        """List of tracked keys that have not been read."""
        return [key for key in self._tracked_keys if (key not in self._read_keys)]
    
    @property
    def all_read(self) -> bool:
        """Check if all keys have been read."""
        return (len(self.unread_keys) == 0) and all(n.all_read for n in self._nested_dicts.values())
    
    @property
    def any_read(self) -> bool:
        """Check if any keys have been read."""
        return (len(self._read_keys) > 0)
    
    @property
    def any_unread(self) -> bool:
        """Check if any keys have been read."""
        return (len(self._read_keys) > 0)
    
    def __str__(self, indent: int = 0) -> str:
        """Print the dependency graph of this TrackedDict and its nested dicts."""
        self._tracking_status = False  # Disable tracking during string generation

        buffer = []
        indent_str = "  " * indent
        yes_str = "\033[92myes\033[0m"  # Green text
        no_str = "\033[91mno\033[0m"    # Red text

        buffer.append(f"{indent_str}{self._context.split('.')[-1]}: all_read={yes_str if self.all_read else no_str}, any_read={yes_str if self.any_read else no_str}, read_keys={self.read_keys}")
        for key, value in self.items():
            if isinstance(value, TrackedDict):
                buffer.append(value.__str__(indent + 1))
            elif (key in self._tracked_keys):
                was_read = yes_str if (key in self._read_keys) else no_str
                buffer.append(f"{indent_str}  {key}: all_read={was_read}, type={type(value)}")
            else:
                buffer.append(f"{indent_str}  \033[90m{key} type={type(value)}\033[0m")
        self._tracking_status = True  # Re-enable tracking

        return "\n".join(buffer)

def load_windio_input(input_data: str | Path | Dict[str, Any]) -> Dict[str, Any]:
    """
    Read a windIO file and return its contents as a dictionary.
    
    Args:
        file_path: Path to the windIO file (YAML or JSON)
    Returns:
        Dictionary representation of the windIO file
    """
    from windIO import load_yaml

    # Import from path
    if isinstance(input_data, str):
        input_data = Path(input_data)

    if isinstance(input_data, Path):
        if input_data.suffix not in [".yaml", ".yml", ".json"]:
            raise ValueError(f"Unsupported windIO file format: '{input_data.suffix}'")
        input_data = load_yaml(input_data)

    # Dictionary input
    if isinstance(input_data, dict):
        return input_data
        
    # Invalid input type
    raise TypeError(f"Invalid input type for windIO file: {type(input_data)}")