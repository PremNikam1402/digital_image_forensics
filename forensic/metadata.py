"""EXIF and Structural Metadata Forensics Module.

Extracts camera make, model, timestamps, software signatures, GPS coordinates,
and exposure settings using ExifTool with automatic fallback to Pillow.
Adheres strictly to forensic standards: no fabricated metadata, handles
missing EXIF gracefully, and does not conflate re-saving with tampering.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, Union
from PIL import Image, ExifTags


COMMON_EDITING_SOFTWARE_KEYWORDS = [
    "photoshop", "gimp", "canva", "lightroom", "paint.net", "snapseed",
    "pixlr", "affinity", "afterlight", "facetune", "procreate", "picsart",
    "vsco", "pixelmator", "coreldraw"
]


def _format_rational(val: Any) -> Optional[float]:
    """Helper to convert Pillow IFDRational or tuple to float."""
    if val is None:
        return None
    try:
        if hasattr(val, "numerator") and hasattr(val, "denominator"):
            return float(val.numerator) / float(val.denominator) if val.denominator != 0 else float(val.numerator)
        if isinstance(val, (tuple, list)) and len(val) == 2:
            return float(val[0]) / float(val[1]) if val[1] != 0 else float(val[0])
        return float(val)
    except Exception:
        return None


def _parse_gps_pillow(gps_info: Dict[int, Any]) -> Optional[Dict[str, Any]]:
    """Parse GPS tags extracted via Pillow into human-readable coordinates."""
    if not gps_info:
        return None
    
    parsed = {}
    gps_tags = {ExifTags.GPSTAGS.get(k, k): v for k, v in gps_info.items()}
    
    def _convert_dms_to_dd(dms, ref):
        try:
            d = _format_rational(dms[0])
            m = _format_rational(dms[1])
            s = _format_rational(dms[2])
            if d is None or m is None or s is None:
                return None
            dd = d + (m / 60.0) + (s / 3600.0)
            if ref in ["S", "W"]:
                dd = -dd
            return round(dd, 6)
        except Exception:
            return None

    lat = gps_tags.get("GPSLatitude")
    lat_ref = gps_tags.get("GPSLatitudeRef", "N")
    lon = gps_tags.get("GPSLongitude")
    lon_ref = gps_tags.get("GPSLongitudeRef", "E")
    alt = gps_tags.get("GPSAltitude")
    alt_ref = gps_tags.get("GPSAltitudeRef", 0)

    if lat and lon:
        lat_dd = _convert_dms_to_dd(lat, lat_ref)
        lon_dd = _convert_dms_to_dd(lon, lon_ref)
        if lat_dd is not None and lon_dd is not None:
            parsed["latitude"] = lat_dd
            parsed["longitude"] = lon_dd
            parsed["coordinates"] = f"{lat_dd}, {lon_dd}"

    if alt is not None:
        alt_val = _format_rational(alt)
        if alt_val is not None:
            parsed["altitude_m"] = round(alt_val, 2)

    return parsed if parsed else None


def _extract_via_pillow(image_input: Union[Path, Image.Image]) -> Dict[str, Any]:
    """Extract metadata using Pillow's native EXIF parser as reliable fallback."""
    extracted = {
        "source": "Pillow (Fallback)",
        "available": False,
        "raw_tags": {},
        "camera_make": None,
        "camera_model": None,
        "date_time": None,
        "software": None,
        "iso": None,
        "exposure_time": None,
        "focal_length": None,
        "orientation": None,
        "dimensions": None,
        "gps": None,
    }

    should_close = False
    try:
        if isinstance(image_input, Image.Image):
            img = image_input
        else:
            img = Image.open(image_input)
            should_close = True

        extracted["dimensions"] = f"{img.width} x {img.height}"
        exif_data = img.getexif()
        
        if not exif_data:
            return extracted

        extracted["available"] = True


        # Map tag IDs to human-readable tag names
        for tag_id, value in exif_data.items():
            tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
            # Store stringified tag value for raw display
            try:
                extracted["raw_tags"][tag_name] = str(value)
            except Exception:
                continue

            if tag_name == "Make":
                extracted["camera_make"] = str(value).strip()
            elif tag_name == "Model":
                extracted["camera_model"] = str(value).strip()
            elif tag_name in ("DateTime", "DateTimeOriginal"):
                extracted["date_time"] = str(value).strip()
            elif tag_name == "Software":
                extracted["software"] = str(value).strip()
            elif tag_name == "Orientation":
                extracted["orientation"] = str(value)

        # Check IFD sub-exif for ISO, Exposure, Focal Length, and GPS
        try:
            for ifd_id in (ExifTags.IFD.Exif, ExifTags.IFD.GPSInfo):
                ifd = exif_data.get_ifd(ifd_id)
                if not ifd:
                    continue
                if ifd_id == ExifTags.IFD.Exif:
                    for sub_id, sub_val in ifd.items():
                        sub_name = ExifTags.TAGS.get(sub_id, str(sub_id))
                        extracted["raw_tags"][sub_name] = str(sub_val)
                        if sub_name == "ISOSpeedRatings" or sub_name == "PhotographicSensitivity":
                            extracted["iso"] = str(sub_val)
                        elif sub_name == "ExposureTime":
                            rat = _format_rational(sub_val)
                            extracted["exposure_time"] = f"1/{round(1/rat)} s" if rat and rat < 1 and rat > 0 else f"{rat} s"
                        elif sub_name == "FocalLength":
                            rat = _format_rational(sub_val)
                            extracted["focal_length"] = f"{rat} mm" if rat else str(sub_val)
                        elif sub_name == "DateTimeOriginal" and not extracted["date_time"]:
                            extracted["date_time"] = str(sub_val)
                elif ifd_id == ExifTags.IFD.GPSInfo:
                    extracted["gps"] = _parse_gps_pillow(dict(ifd))
        except Exception:
            pass


    except Exception:
        pass
    finally:
        if should_close:
            try:
                img.close()
            except Exception:
                pass

    return extracted



def _extract_via_exiftool(image_path: Path) -> Optional[Dict[str, Any]]:
    """Extract full EXIF/XMP/IPTC metadata using ExifTool CLI if installed."""
    exiftool_exe = shutil.which("exiftool")
    if not exiftool_exe:
        return None

    try:
        cmd = [exiftool_exe, "-json", "-G", str(image_path)]
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10,
            check=True
        )
        data = json.loads(result.stdout)
        if not data or not isinstance(data, list):
            return None
        
        info = data[0]
        extracted = {
            "source": "ExifTool CLI",
            "available": True,
            "raw_tags": {k: str(v) for k, v in info.items()},
            "camera_make": info.get("EXIF:Make") or info.get("Make"),
            "camera_model": info.get("EXIF:Model") or info.get("Model"),
            "date_time": (
                info.get("EXIF:DateTimeOriginal")
                or info.get("DateTimeOriginal")
                or info.get("EXIF:CreateDate")
                or info.get("CreateDate")
            ),
            "software": info.get("EXIF:Software") or info.get("Software"),
            "iso": str(info.get("EXIF:ISO") or info.get("ISO") or ""),
            "exposure_time": str(info.get("EXIF:ExposureTime") or info.get("ExposureTime") or ""),
            "focal_length": str(info.get("EXIF:FocalLength") or info.get("FocalLength") or ""),
            "orientation": str(info.get("EXIF:Orientation") or info.get("Orientation") or ""),
            "dimensions": (
                f"{info.get('File:ImageWidth', '')} x {info.get('File:ImageHeight', '')}"
                if info.get("File:ImageWidth")
                else None
            ),
            "gps": None,
        }

        lat = info.get("Composite:GPSLatitude") or info.get("GPSLatitude")
        lon = info.get("Composite:GPSLongitude") or info.get("GPSLongitude")
        alt = info.get("Composite:GPSAltitude") or info.get("GPSAltitude")
        if lat and lon:
            extracted["gps"] = {
                "latitude": lat,
                "longitude": lon,
                "coordinates": f"{lat}, {lon}",
                "altitude_m": alt,
            }

        return extracted
    except Exception:
        return None


def extract_metadata(image_path_or_obj: Union[str, Path, Image.Image]) -> Dict[str, Any]:
    """Extract image metadata, handling missing EXIF and software indicators.
    
    Returns structured forensic dictionary with metadata status and analysis.
    """
    tmp_path: Optional[Path] = None

    try:
        if isinstance(image_path_or_obj, Image.Image):
            # If ExifTool is present, write temporary JPEG preserving EXIF bytes
            if shutil.which("exiftool"):
                import tempfile
                tfile = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
                tfile.close()
                try:
                    exif_bytes = image_path_or_obj.getexif()
                    image_path_or_obj.save(tfile.name, format="JPEG", exif=exif_bytes)
                except Exception:
                    image_path_or_obj.save(tfile.name, format="JPEG")
                target_path = Path(tfile.name)
                tmp_path = target_path
                data = _extract_via_exiftool(target_path)
            else:
                data = None

            if not data:
                data = _extract_via_pillow(image_path_or_obj)
        else:
            target_path = Path(image_path_or_obj)
            data = _extract_via_exiftool(target_path)
            if not data:
                data = _extract_via_pillow(target_path)

        if not data:
            data = {
                "source": "None",
                "available": False,
                "raw_tags": {},
                "camera_make": None,
                "camera_model": None,
                "date_time": None,
                "software": None,
                "iso": None,
                "exposure_time": None,
                "focal_length": None,
                "orientation": None,
                "dimensions": None,
                "gps": None,
            }

        # Forensic Analysis of Software Signature
        software_str = (data.get("software") or "").lower()
        editing_software_detected = any(tool in software_str for tool in COMMON_EDITING_SOFTWARE_KEYWORDS)
        
        has_camera_info = bool(data.get("camera_make") or data.get("camera_model"))
        has_exif = bool(data.get("available") and len(data.get("raw_tags", {})) > 0)

        # Forensic interpretation commentary
        if not has_exif:
            forensic_summary = "Metadata unavailable. The image has no EXIF tags (common for screenshots, social media downloads, or stripped files)."
            status_text = "Metadata unavailable"
        elif editing_software_detected:
            forensic_summary = (
                f"Editing software signature detected: '{data.get('software')}'. "
                "Notice: Software signatures indicate export from an image editor but do NOT "
                "conclusively prove malicious content tampering without pixel-level corroboration."
            )
            status_text = "Software Signature Present"
        elif has_camera_info:
            forensic_summary = (
                f"Camera hardware signature verified: {data.get('camera_make')} {data.get('camera_model')}. "
                "Consistent with direct capture device metadata."
            )
            status_text = "Authentic Camera Hardware Tags Found"
        else:
            forensic_summary = "Partial metadata present; no editing software signatures detected."
            status_text = "Partial Metadata Available"

        return {
            "status": status_text,
            "has_exif": has_exif,
            "source": data.get("source", "Unknown"),
            "camera_make": data.get("camera_make") or "Not recorded",
            "camera_model": data.get("camera_model") or "Not recorded",
            "date_time": data.get("date_time") or "Not recorded",
            "software": data.get("software") or "Not recorded",
            "iso": data.get("iso") or "Not recorded",
            "exposure_time": data.get("exposure_time") or "Not recorded",
            "focal_length": data.get("focal_length") or "Not recorded",
            "orientation": data.get("orientation") or "Not recorded",
            "dimensions": data.get("dimensions") or "Not recorded",
            "gps": data.get("gps"),
            "raw_tags": data.get("raw_tags", {}),
            "editing_software_detected": editing_software_detected,
            "forensic_summary": forensic_summary,
        }

    finally:
        if tmp_path and tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass

