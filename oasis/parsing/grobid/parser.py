"""Парсер TEI XML ответов от GROBID."""

from typing import Any

from lxml import etree
from loguru import logger

from oasis.parsing.grobid.xml_parser import parse_bibl_struct

GROBID_NAMESPACE = "http://www.tei-c.org/ns/1.0"
GROBID_NSMAP = {"tei": GROBID_NAMESPACE}


def parse_grobid_response(tei_xml: str) -> list[dict[str, Any]]:
    """Парсит TEI XML ответ от GROBID.

    Args:
        tei_xml: TEI XML строка

    Returns:
        Список словарей с полями ссылок
    """
    try:
        root = etree.fromstring(tei_xml.encode("utf-8"))
        references = []
        ref_divs = root.xpath(
            ".//tei:div[@type='references']//tei:listBibl//tei:biblStruct | "
            ".//tei:listBibl//tei:biblStruct",
            namespaces=GROBID_NSMAP,
        )

        ref_counter = 1
        for bibl_struct in ref_divs:
            ref = parse_bibl_struct(bibl_struct)
            if ref:
                ref_n = bibl_struct.get("n") or ref_counter
                try:
                    ref["ref_number"] = (
                        int(ref_n) if isinstance(ref_n, str) and ref_n.isdigit() else ref_counter
                    )
                except (ValueError, AttributeError):
                    ref["ref_number"] = ref_counter
                ref_counter += 1
                references.append(ref)

        return references
    except etree.XMLSyntaxError as e:
        logger.error(f"Ошибка парсинга TEI XML: {e}")
        return []
    except Exception as e:
        logger.error(f"Неожиданная ошибка при парсинге GROBID: {e}")
        return []

