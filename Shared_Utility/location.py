import asyncio
import sys

# --- CONFIGURATION ---
try:
    import winsdk.windows.devices.geolocation as wdg
except ImportError:
    print("❌ CRITICAL: 'winsdk' library not found.")
    print("   Please run: pip install winsdk")
    sys.exit(1)

async def _get_precise_coords():
    """
    Asks Windows for High-Accuracy GPS coordinates.
    """
    try:
        locator = wdg.Geolocator()
        
        status = locator.location_status
        if status == wdg.PositionStatus.DISABLED:
            return None, "Location Services are OFF in Settings."
        
        # Request Position (High Accuracy)
        pos = await locator.get_geoposition_async()
        
        lat = pos.coordinate.point.position.latitude
        lon = pos.coordinate.point.position.longitude
        
        return (lat, lon), "SUCCESS"

    except Exception as e:
        return None, str(e)

def get_current_location():
    """
    Sync Wrapper for the GPS function.
    """
    try:
        # Run the async GPS request synchronously
        coords, msg = asyncio.run(_get_precise_coords())

        if coords:
            lat, lon = coords
            
            # --- FIXED LINK FORMAT ---
            # This is the standard format that works on all phones and browsers
            maps_link = f"https://www.google.com/maps?q={lat},{lon}"
            # -------------------------
            
            return {
                "status": "SUCCESS",
                "latitude": lat,
                "longitude": lon,
                "maps_url": maps_link,
                "message": "Device Location Found"
            }
        else:
            return {
                "status": "FAIL",
                "message": f"GPS Failed: {msg}",
                "maps_url": ""
            }

    except Exception as e:
        return {
            "status": "ERROR",
            "message": f"System Error: {e}",
            "maps_url": ""
        }

# --- TESTING BLOCK ---
if __name__ == "__main__":
    print("\n-----------------------------------")
    print(" 📡 REQUESTING DEVICE GPS...")
    print(" ⏳ Waiting for sensor lock...")
    print("-----------------------------------")
    
    loc = get_current_location()
    
    if loc["status"] == "SUCCESS":
        print(f" ✅ GPS LOCKED.")
        print(f" 📍 Latitude:  {loc['latitude']}")
        print(f" 📍 Longitude: {loc['longitude']}")
        print(f" 🔗 Map: {loc['maps_url']}")  # Try clicking this new link!
    else:
        print(f" ❌ LOCATION FAILED.")
        print(f" ⚠️ Reason: {loc['message']}")
    
    print("-----------------------------------")