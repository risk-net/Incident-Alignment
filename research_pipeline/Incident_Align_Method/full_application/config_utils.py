#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import configparser
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


BASE_DIR = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = BASE_DIR / "config" / "Incident_Align_Method-full_application-config.ini"


def load_config(config_path: Optional[str] = None) -> configparser.ConfigParser:
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    parser = configparser.ConfigParser()
    parser.read(path, encoding="utf-8")
    return parser


def require_section(parser: configparser.ConfigParser, section: str) -> configparser.SectionProxy:
    if section not in parser:
        raise KeyError(f"配置文件缺少 [{section}] 段")
    return parser[section]


def get_required(section: configparser.SectionProxy, key: str) -> str:
    value = (section.get(key, "") or "").strip()
    if not value:
        raise ValueError(f"配置项缺失: [{section.name}] {key}")
    return value


def get_bool(section: configparser.SectionProxy, key: str, fallback: bool) -> bool:
    if key not in section:
        return fallback
    return section.getboolean(key)


def get_int(section: configparser.SectionProxy, key: str, fallback: int) -> int:
    if key not in section:
        return fallback
    return section.getint(key)


def get_float(section: configparser.SectionProxy, key: str, fallback: float) -> float:
    if key not in section:
        return fallback
    return section.getfloat(key)


def resolve_path(path_value: str, base_dir: Optional[Path] = None) -> str:
    raw = str(path_value).strip()
    if not raw:
        raise ValueError("路径不能为空")
    if os.path.isabs(raw):
        return raw
    root = base_dir or BASE_DIR
    return str((root / raw).resolve())


def split_csv(raw_value: str) -> Iterable[str]:
    return [item.strip() for item in str(raw_value).split(",") if item.strip()]


def load_database_config(parser: configparser.ConfigParser, section_name: str = "Database") -> Dict[str, Any]:
    section = require_section(parser, section_name)
    password_env = get_required(section, "password_env")
    password = os.environ.get(password_env, "")
    if not password:
        raise ValueError(
            f"数据库密码环境变量未设置: {password_env}. "
            f"请先导出该环境变量，再运行脚本。"
        )
    return {
        "host": get_required(section, "host"),
        "port": get_int(section, "port", 5432),
        "database": get_required(section, "database"),
        "user": get_required(section, "user"),
        "password": password,
        "password_env": password_env,
        "classification_result": section.get("classification_result", "AIrisk_relevant_event").strip() or "AIrisk_relevant_event",
        "source_relation": section.get("source_relation", "v_alignment_input_v1").strip() or "v_alignment_input_v1",
    }
