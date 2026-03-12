import asyncio              # Run async GPS calls
import sys                  # Exit program on critical failure

# --- CONFIGURATION ---
try:
    # Windows SDK for accessing system GPS
    import winsdk.windows.devices.geolocation as wdg
except ImportError:
    # winsdk is mandatory for this module
    print("❌ CRITICAL: 'winsdk' library not found.")
    print("   Please run: pip install winsdk")
    sys.exit(1)

async def _get_precise_coords():
    """
    Requests high-accuracy GPS coordinates from Windows Location Services.
    """
    try:
        locator = wdg.Geolocator()

        # Check whether location services are enabled
        status = locator.location_status
        if status == wdg.PositionStatus.DISABLED:
            return None, "Location Services are OFF in Settings."

        # Asynchronously request current device position
        pos = await locator.get_geoposition_async()

        lat = pos.coordinate.point.position.latitude
        lon = pos.coordinate.point.position.longitude

        return (lat, lon), "SUCCESS"

    except Exception as e:
        # Catch system / permission / sensor errors
        return None, str(e)

def get_current_location():
    """
    Synchronous wrapper around the async GPS request.
    Makes the function easy to use in normal scripts.
    """
    try:
        # Execute async GPS call in a blocking manner
        coords, msg = asyncio.run(_get_precise_coords())

        if coords:
            lat, lon = coords

            # Universal Google Maps link (works on all devices)
            maps_link = f"https://www.google.com/maps?q={lat},{lon}"

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
        # Fallback for unexpected runtime issues
        return {
            "status": "ERROR",
            "message": f"System Error: {e}",
            "maps_url": ""
        }

# --- TESTING / DEMO MODE ---
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
        print(f" 🔗 Map: {loc['maps_url']}")
    else:
        print(f" ❌ LOCATION FAILED.")
        print(f" ⚠️ Reason: {loc['message']}")

    print("-----------------------------------")
