# Capture Protocol — One-Page Non-Engineer Guide

**Tool required:** 3D Scanner App (free, App Store — install on iPhone 15 Pro or newer for LiDAR)  
**Time per room:** approximately 2-5 minutes  
**What you'll get:** a floor plan of your property in about 15 minutes total

---

## Before You Start

1. Install **3D Scanner App** from the App Store (it is free; no account needed)
2. Open the app and tap **New Scan → LiDAR mode** (this uses the dot sensor on the back of your phone)
3. Make sure the space is reasonably lit — open blinds if possible
4. Clear the floor of trip hazards; you will be walking slowly

---

## LiDAR Scan (best accuracy, iPhone 15 Pro / Pro Max only)

1. Stand at one corner of the room, hold the phone at **hip height**, camera facing forward
2. Walk slowly clockwise around the room's perimeter, staying 0.5-1 metre from the walls
3. At each corner, **pause for 2 seconds** and tilt the phone slowly up then down to capture the full wall height
4. At each door or window opening, pause and scan across it
5. At doorways connecting to other rooms: take **3 photos** that show both rooms at once  
   *(hold still, tap the shutter — this is important for joining rooms together)*
6. Return to your starting corner and pause for 2 seconds (this closes the loop)
7. Tap **Export → Point Cloud → PLY** — save to Files → share to your laptop

Repeat steps 1–7 for each room. Export each room as a **separate PLY file** and put them in a folder:

```
scan/
  living_room.ply
  bedroom.ply
  kitchen.ply
```

---

## Video Scan (any iPhone 15 or newer)

1. Open the **native Camera app**, switch to **Video**, set to 1080p 30fps
2. Walk the same clockwise perimeter route as the LiDAR scan
3. Hold the phone at hip height, move smoothly — no sudden jerks
4. At each corner, pause for 3-4 seconds
5. At doorways between rooms, slow down and show the doorway clearly
6. Export the video file (.MOV) and hand to the pipeline

---

## Photo Scan (any iPhone 15 or newer, lowest accuracy)

Per room, take **6-8 photos**:
- 2 photos of each wall (one from each corner of the opposite wall)
- 1 photo from the centre of the room looking at each of the 4 walls
- At doorways: **3 photos showing the door AND the adjacent room** — this is essential

Put photos in per-room folders:

```
photos/
  living_room/
    img_001.jpg … img_008.jpg
  bedroom/
    img_001.jpg … img_006.jpg
```

---

## Handing Files to the Pipeline

```bash
# LiDAR (PLY)
python run.py --input scan/ --tier lidar

# Video
python run.py --input walkthrough.mov --tier video

# Photos
python run.py --input photos/ --tier photo
```

Results appear in `output/<capture_id>/` — a JSON file and an SVG floor plan.

---

## What To Avoid

- Don't wave or shake the phone
- Don't scan through mirrors or glass doors (they confuse depth sensors)
- Don't scan in complete darkness
- Don't skip the doorway transitional photos — they are how rooms connect
- If the scan seems stuck, stop, re-open the app and start again from a corner
