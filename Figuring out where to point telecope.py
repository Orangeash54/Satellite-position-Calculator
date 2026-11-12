
from skyfield.api import load
from skyfield.iokit import parse_tle_file
from skyfield.api import wgs84
import matplotlib.pyplot as plt
import numpy as np

ts = load.timescale()

#Loading the TLE file of the space stations

with load.open('GPS Satellites.txt') as f:
    satellites = list(parse_tle_file(f, ts))

print('Loaded', len(satellites), 'satellites')

# Position of the Marsh Observatory at Warwick

latitude = 52.38
longitude = 1.56
elevation = 86

bluffdon = wgs84.latlon(latitude, longitude, elevation)

# Bluffdon is our position and wgs84 converts latitude, longitude and elevation into position vectors

eph = load("de421.bsp")

# code for determinging the sun ephemeris

moon_exclusion_angle = 5.0
observer = eph['earth'] + bluffdon

t_now = ts.now()

centre_time = 60.0

t = t_now + centre_time / 86400

# Divide by 86400 as skyfeild works in day long units of time

moon_position = observer.at(t).observe(eph['moon']).apparent()

# Moon position at t = 60 seconds

search_step = 0.1

search_offsets = np.arange(
    centre_time - 60,
    centre_time + 60 + search_step,
    search_step
)

search_times = t_now + search_offsets / 86400

centre_index = int(round(centre_time / search_step))

# Function to convert seconds into readable UTC time

def format_time(seconds):

    return (
        t_now + seconds / 86400
    ).utc_datetime().strftime('%H:%M:%S.%f UTC')

def calculate_observation_times(satellite, t_now, centre_time):

    # Field of view dimensions in arcminutes
    half_width = 16.0 * 60
    half_height = 10.5 * 60

    # Search times around the centre
    search_step = 0.1

    offsets = np.arange(
        centre_time - 60,
        centre_time + 60 + search_step,
        search_step
    )

    times = t_now + offsets / 86400

    # Satellite position at centre time
    centre_position = (satellite - bluffdon).at(
        t_now + centre_time / 86400
    )

    centre_vector = centre_position.position.au
    centre_vector /= np.linalg.norm(centre_vector)

    # Centre RA and DEC
    ra = np.arctan2(centre_vector[1], centre_vector[0]) % (2 * np.pi)
    dec = np.arcsin(centre_vector[2])

    # East and north directions
    east = np.array([-np.sin(ra), np.cos(ra), 0])

    north = np.array([
        -np.sin(dec) * np.cos(ra),
        -np.sin(dec) * np.sin(ra),
        np.cos(dec)
    ])

    # Satellite positions throughout search
    positions = (satellite - bluffdon).at(times).position.au

    positions /= np.linalg.norm(positions, axis=0)

    # Position relative to field centre (arcseconds)
    denominator = centre_vector @ positions

    x = np.arctan2(east @ positions, denominator) * 206264.806
    y = np.arctan2(north @ positions, denominator) * 206264.806

    # 206264.806 comes from converting radians into arcseconds

    # Check whether satellite is inside the field
    inside = (
        (np.abs(x) <= half_width)
        & (np.abs(y) <= half_height)
    )

    centre_index = int(round(centre_time / search_step))

    if not inside[centre_index]:
        return None

    # Find entry and exit indices
    left = centre_index
    right = centre_index

    while left > 0 and inside[left - 1]:
        left -= 1

    while right < len(inside) - 1 and inside[right + 1]:
        right += 1

    # Calculate entry and exit times
    entry = offsets[left] if left > 0 else None
    exit_time = offsets[right] if right < len(inside) - 1 else None

    return ra, dec, entry, exit_time


visible_satellites = []

for satellite in satellites:

    # Satellite position at t = 60 seconds

    t = t_now + 60 / 86400

    position = (satellite - bluffdon).at(t)

    alt, az, distance = position.altaz()

    # Check sunlight

    sunlit = position.is_sunlit(eph)

    # Check Moon separation

    separation = position.separation_from(moon_position)

    # Apply filtering conditions

    if (
        alt.degrees > 35
        and sunlit
        and separation.degrees > moon_exclusion_angle
    ):

        # Calculate RA, DEC and observation times

        result = calculate_observation_times(
            satellite, t_now, 60
        )

        if result is not None:

            ra, dec, entry, exit_time = result

            visible_satellites.append({
                "name": satellite.name,
                "RA": np.degrees(ra) / 15,
                "DEC": np.degrees(dec),
                "entry": entry,
                "centre": 60,
                "exit": exit_time
            })


# Print results

print("\nGPS SATELLITE OBSERVATION TIMES")
print("Number of suitable satellites:", len(visible_satellites))

for sat in visible_satellites:

    print("\n-----------------------------")

    print("Satellite:", sat["name"])
    print(f"RA: {sat['RA']:.8f} hours")
    print(f"DEC: {sat['DEC']:.8f} degrees")

    if sat["entry"] is not None:
        print("Enters field:", format_time(sat["entry"]))
    else:
        print("Entry: Before search interval")

    print("Crosses centre:", format_time(sat["centre"]))

    if sat["exit"] is not None:
        print("Leaves field:", format_time(sat["exit"]))
    else:
        print("Exit: After search interval")