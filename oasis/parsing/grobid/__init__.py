"""GROBID клиент и парсеры для извлечения ссылок из PDF."""

from oasis.parsing.grobid.client import GrobidClient
from oasis.parsing.grobid.parser import parse_grobid_response
from oasis.parsing.grobid.xml_parser import parse_bibl_struct

__all__ = ["GrobidClient", "parse_grobid_response", "parse_bibl_struct"]

