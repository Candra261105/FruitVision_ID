import os
from pathlib import Path
from playwright.sync_api import sync_playwright

SAMPLE_PREDICTION = {
    "model_available": True,
    "mode": "custom",
    "message": "Terdeteksi 3 objek pada citra dengan pencahayaan seimbang.",
    "limitation": "Evaluasi awal 3 kelas.",
    "original_image": "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='400' height='300'%3E%3Crect width='400' height='300' fill='%23182822'/%3E%3Ccircle cx='120' cy='150' r='60' fill='%23d64530'/%3E%3Ccircle cx='260' cy='140' r='55' fill='%23e5a93c'/%3E%3C/svg%3E",
    "enhanced_image": "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='400' height='300'%3E%3Crect width='400' height='300' fill='%231d322b'/%3E%3Ccircle cx='120' cy='150' r='60' fill='%23e6523c'/%3E%3Ccircle cx='260' cy='140' r='55' fill='%23f3bc56'/%3E%3C/svg%3E",
    "enhancement_status": "enhanced",
    "enhancement_steps": ["gamma 0.85", "contrast stretch (p5-p95)", "median denoise 3x3"],
    "quality": {
        "width": 1920,
        "height": 1080,
        "brightness": 124,
        "contrast": 68,
        "noise": 0.42,
        "glare_ratio": 0.02,
        "blur_score": 185.4,
        "warnings": []
    },
    "detections": [
        {
            "id": 1,
            "class_name": "tomato",
            "fruit": "tomat",
            "confidence": 0.94,
            "box": {"x1": 0.15, "y1": 0.25, "x2": 0.45, "y2": 0.75},
            "stage": "matang",
            "stage_available": True,
            "review_status": "accepted",
            "warnings": [],
            "conditions": {
                "maturity": {"task": "maturity", "status": "available", "label": "ripe", "label_display": "Matang optimal", "confidence": 0.92, "detail": "Pewarnaan merah merata >85%"},
                "rot": {"task": "rot", "status": "available", "label": "fresh", "label_display": "Segar (tidak busuk)", "confidence": 0.96, "detail": "Kulit buah utuh tanpa lesi"}
            },
            "crop_image": "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='100' height='100'%3E%3Crect width='100' height='100' fill='%23223830'/%3E%3Ccircle cx='50' cy='50' r='38' fill='%23e6523c'/%3E%3C/svg%3E",
            "color_indicator": {
                "hue_segments": {"merah": 72.5, "kuning": 18.2, "hijau": 6.1, "lainnya": 3.2},
                "saturation": 0.82,
                "value": 0.78,
                "status": "available",
                "detail": "Dominan rona merah cerah."
            }
        },
        {
            "id": 2,
            "class_name": "pumpkin",
            "fruit": "labu",
            "confidence": 0.88,
            "box": {"x1": 0.52, "y1": 0.20, "x2": 0.85, "y2": 0.72},
            "stage": "matang",
            "stage_available": True,
            "review_status": "accepted",
            "warnings": [],
            "conditions": {
                "maturity": {"task": "maturity", "status": "available", "label": "ripe", "label_display": "Matang", "confidence": 0.86, "detail": "Tekstur kulit mengeras"}
            },
            "crop_image": "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='100' height='100'%3E%3Crect width='100' height='100' fill='%23223830'/%3E%3Ccircle cx='50' cy='50' r='38' fill='%23f3bc56'/%3E%3C/svg%3E",
            "color_indicator": {
                "hue_segments": {"merah": 12.0, "kuning": 74.0, "hijau": 8.5, "lainnya": 5.5},
                "saturation": 0.76,
                "value": 0.84,
                "status": "available",
                "detail": "Dominan rona oranye/kuning."
            }
        }
    ],
    "capabilities": {
        "maturity": {"status": "available", "detail": "Model kematangan aktif untuk tomat dan labu."},
        "rot": {"status": "available", "detail": "Deteksi kebusukan permukaan aktif."},
        "disease": {"status": "review_required", "detail": "Data bercak daun memerlukan verifikasi manual."},
        "grade": {"status": "unavailable", "detail": "Estimasi grade berat/diameter belum dikalibrasi."}
    },
    "recommendations": [
        "Tomat objek 01 siap petik/distribusi ke pasar retail dalam 1-2 hari.",
        "Labu objek 02 siap panen dengan tangkai dipotong menyisakan 3-5 cm.",
        "Simpan sampel pada suhu ruang 15-20°C dengan sirkulasi udara baik."
    ],
    "ai": {
        "status": "available",
        "detail": "Penjelasan agronomis berbasis model inferensi lokal.",
        "summary": "Sampel menunjukkan tingkat kematangan komersial yang seragam tanpa indikasi busuk basah.",
        "recommendations": ["Pisahkan buah dengan bercak minor sebelum pengemasan."],
        "uncertainties": ["Estimasi ketebalan kulit labu hanya didasarkan pada penampakan luar."],
        "disclaimer": "Rekomendasi bersifat asisten agronomis pendukung, bukan jaminan mutu mutlak."
    },
    "inspection_id": "insp_20261125_sample"
}

def run_visual_verification():
    out_dir = Path("test-results/ui-verification")
    out_dir.mkdir(parents=True, exist_ok=True)
    executable = os.getenv(
        "FRUITVISION_CHROMIUM",
        str(Path(os.environ["LOCALAPPDATA"]) / "ms-playwright/chromium-1255/chrome-win64/chrome.exe"),
    )

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True, executable_path=executable)
        
        # Test 1: Desktop Viewport (1280px)
        context = browser.new_context(viewport={"width": 1280, "height": 850})
        page = context.new_page()
        
        # Mock backend responses so we can test full states cleanly
        def handle_route(route):
            url = route.request.url
            if "/api/model-info" in url:
                route.fulfill(
                    status=200,
                    content_type="application/json",
                    body='{"name":"yolo26n-adaptive.pt","mode":"custom","model_available":true,"classes":["tomato","pumpkin","lemon"],"stage_classes":["mentah","setengah_matang","matang"],"limitation":"Model 3 kelas terkalibrasi."}'
                )
            elif "/api/predict/image" in url:
                import json
                route.fulfill(status=200, content_type="application/json", body=json.dumps(SAMPLE_PREDICTION))
            else:
                route.continue_()

        page.route("**/api/**", handle_route)
        page.goto("http://127.0.0.1:5180", wait_until="networkidle")
        
        # Trigger an image upload to see populated state
        # Create a tiny dummy file
        dummy_file = Path("test-results/dummy.jpg")
        dummy_file.write_bytes(b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a\x1f\x1e\x1d\x1a\x1c\x1c $.' \",#\x1c\x1c(7),01444\x1f'9=82<.342\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00\xff\xc4\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\xff\xda\x00\x08\x01\x01\x00\x00?\x00\xbf\x00\xff\xd9")
        
        page.set_input_files("#fruit-upload", str(dummy_file))
        page.wait_for_timeout(500)
        
        # Capture Tab 1: Deteksi & Objek
        page.screenshot(path=str(out_dir / "01_desktop_tab_detections.png"), full_page=True)
        print("Captured: 01_desktop_tab_detections.png")
        
        # Click Tab 2: Kualitas Citra
        page.get_by_role("tab", name="Kualitas Citra").click()
        page.wait_for_timeout(300)
        page.screenshot(path=str(out_dir / "02_desktop_tab_quality.png"), full_page=True)
        print("Captured: 02_desktop_tab_quality.png")
        
        # Click Tab 3: Kondisi & AI
        page.get_by_role("tab", name="Kondisi & AI").click()
        page.wait_for_timeout(300)
        page.screenshot(path=str(out_dir / "03_desktop_tab_analysis.png"), full_page=True)
        print("Captured: 03_desktop_tab_analysis.png")
        
        # Test 2: Tablet Viewport (768px)
        tablet_context = browser.new_context(viewport={"width": 768, "height": 1024})
        tablet_page = tablet_context.new_page()
        tablet_page.route("**/api/**", handle_route)
        tablet_page.goto("http://127.0.0.1:5180", wait_until="networkidle")
        tablet_page.set_input_files("#fruit-upload", str(dummy_file))
        tablet_page.wait_for_timeout(500)
        tablet_page.screenshot(path=str(out_dir / "04_tablet_768px.png"), full_page=True)
        print("Captured: 04_tablet_768px.png")
        
        # Test 3: Mobile Viewport (375px)
        mobile_context = browser.new_context(viewport={"width": 375, "height": 812})
        mobile_page = mobile_context.new_page()
        mobile_page.route("**/api/**", handle_route)
        mobile_page.goto("http://127.0.0.1:5180", wait_until="networkidle")
        mobile_page.set_input_files("#fruit-upload", str(dummy_file))
        mobile_page.wait_for_timeout(500)
        mobile_page.screenshot(path=str(out_dir / "05_mobile_375px.png"), full_page=True)
        print("Captured: 05_mobile_375px.png")
        
        browser.close()

if __name__ == "__main__":
    run_visual_verification()
