#!/usr/bin/env python3
"""Display node: fullscreen window on the participant's monitor (Tkinter), plus the keypad.

Sub:  /display       (String, JSON from experiment_node)
        {"screen": "text" | "reach" | "score" | "rating" | "countdown" | "result" | "final",
         "title", "text", "cubes": [colour, ...] left to right as the participant sees them,
         "choice": index or -1, "target": index (result), "number" (countdown),
         "points" (result, final), "hud": {"round", "rounds", "total"}, "focus": bool}
        A message with "screen" replaces the screen; one without (e.g. {"choice": 2}) updates it.
Pub:  /rating        (std_msgs/Int32)  keys 1-7 (number row or keypad) while the window has focus
      /button_press  (std_msgs/Int32)  keys a s d f -> 0-3, only with display.keyboard_buttons
                                       (dry runs without the push buttons)
Esc leaves fullscreen. display.geometry (e.g. "+1920+0") puts the window on the participant's
monitor. "focus": true takes the keyboard focus, so the game keys reach this window.
"""
import json
import queue
import tkinter as tk

import rospy
from std_msgs.msg import Int32, String

COLORS = {"orange": "#ff8000", "red": "#e00000", "yellow": "#ffd800", "green": "#00b000",
          "blue": "#1060ff", "purple": "#9020c0", "white": "#ffffff"}
BG, FG, DIM, GOLD = "#101010", "#f0f0f0", "#808080", "#ffc400"
GOOD, BAD = "#40d040", "#ff5050"
EMPTY = {"screen": "text", "title": "", "text": "", "cubes": [], "choice": -1, "target": -1,
         "number": None, "points": None, "hud": None}
BUTTON_KEYS = "asdf"


class Display:
    def __init__(self):
        cfg = rospy.get_param("~display", {})
        self.messages = queue.Queue()
        self.rating_pub = rospy.Publisher("/rating", Int32, queue_size=10)
        self.button_pub = (rospy.Publisher("/button_press", Int32, queue_size=10)
                           if cfg.get("keyboard_buttons", False) else None)
        rospy.Subscriber("/display", String, lambda m: self.messages.put(m.data))

        self.root = tk.Tk()
        self.root.title("HRI experiment")
        self.root.configure(bg=BG)
        if cfg.get("geometry"):
            self.root.geometry(cfg["geometry"])
            self.root.update_idletasks()
        self.root.attributes("-fullscreen", bool(cfg.get("fullscreen", True)))
        self.root.bind("<Escape>", lambda e: self.root.attributes("-fullscreen", False))
        self.root.bind("<Key>", self.on_key)
        self.canvas = tk.Canvas(self.root, bg=BG, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda e: self.draw())
        self.keys = BUTTON_KEYS.upper() if self.button_pub else ""
        self.state = dict(EMPTY, title="Please wait")
        self.pressed = None
        self.root.after(50, self.poll)

    def on_key(self, event):
        key = event.char
        if key and key in "1234567":
            self.rating_pub.publish(Int32(int(key)))
            if self.state["screen"] == "rating":
                self.pressed = int(key)
                self.draw()
        elif self.button_pub and key and key in BUTTON_KEYS:
            self.button_pub.publish(Int32(BUTTON_KEYS.index(key)))

    def poll(self):
        if rospy.is_shutdown():
            self.root.destroy()
            return
        changed = False
        while not self.messages.empty():
            msg = json.loads(self.messages.get_nowait())
            if "screen" in msg:
                self.state = dict(EMPTY)
                self.pressed = None
            self.state.update(msg)
            if msg.get("focus"):
                self.root.focus_force()
            changed = True
        if changed:
            self.draw()
        self.root.after(50, self.poll)

    def draw(self):
        c, s = self.canvas, self.state
        c.delete("all")
        w, h = c.winfo_width(), c.winfo_height()
        big = max(24, h // 12)
        screen = s["screen"]
        title_color = {"result": GOOD if s.get("choice", -1) == s.get("target") else BAD}.get(screen, FG)
        c.create_text(w / 2, h * 0.15, text=s.get("title", ""), fill=title_color,
                      font=("Helvetica", big, "bold"))
        if s.get("hud"):
            self.draw_hud(w, h, s["hud"])
        if screen in ("reach", "score", "countdown", "result") and s.get("cubes"):
            self.draw_cubes(w, h, s["cubes"], s.get("choice", -1),
                            s.get("target", -1) if screen == "result" else -1, dim=screen == "countdown")
        elif screen == "rating":
            self.draw_scale(w, h)
        if screen == "countdown" and s.get("number") is not None:
            c.create_text(w / 2, h * 0.2, text=str(s["number"]), fill=GOLD,
                          font=("Helvetica", int(h // 5), "bold"))
        if screen in ("result", "final") and s.get("points") is not None:
            text = "+%d points" % s["points"] if screen == "result" else str(s["points"])
            c.create_text(w / 2, h * (0.28 if screen == "result" else 0.45), text=text,
                          fill=GOLD, font=("Helvetica", int(h // (10 if screen == "result" else 4)), "bold"))
        if s.get("text"):
            c.create_text(w / 2, h * 0.82, text=s["text"], fill=FG, width=w * 0.8,
                          font=("Helvetica", max(16, h // 28)), justify="center")

    def draw_hud(self, w, h, hud):
        font = ("Helvetica", max(14, h // 30), "bold")
        self.canvas.create_text(w * 0.03, h * 0.04, anchor="nw", fill=DIM, font=font,
                                text="Round %d / %d" % (hud["round"], hud["rounds"]))
        self.canvas.create_text(w * 0.97, h * 0.04, anchor="ne", fill=GOLD, font=font,
                                text="Score %d" % hud["total"])

    def draw_cubes(self, w, h, cubes, choice, target=-1, dim=False):
        n = len(cubes)
        size = min(w / (n * 1.6), h * 0.3)
        gap = (w - n * size) / (n + 1)
        y0 = h * 0.38
        small = ("Helvetica", max(14, h // 30), "bold")
        for i, color in enumerate(cubes):
            x0 = gap + i * (size + gap)
            chosen = i == choice
            faded = dim or (choice >= 0 and not chosen and i != target)
            outline, width = (GOLD, 12) if i == target else (FG, 12) if chosen else (BG, 1)
            self.canvas.create_rectangle(x0, y0, x0 + size, y0 + size, fill=COLORS.get(color, color),
                                         outline=outline, width=width, stipple="gray50" if faded else "")
            label = "ROBOT'S CUBE" if i == target else "YOUR CHOICE" if chosen else ""
            if self.keys and i < len(self.keys):
                self.canvas.create_text(x0 + size / 2, y0 + size / 2, text=self.keys[i],
                                        fill=BG, font=("Helvetica", int(size / 3), "bold"))
            if label:
                self.canvas.create_text(x0 + size / 2, y0 + size + h * 0.05, text=label,
                                        fill=GOLD if i == target else FG, font=small)
            if i == target and chosen:
                self.canvas.create_text(x0 + size / 2, y0 + size + h * 0.10, text="YOUR CHOICE",
                                        fill=FG, font=small)

    def draw_scale(self, w, h):
        size = min(w / 11, h * 0.15)
        gap = (w - 7 * size) / 8
        y0 = h * 0.45
        for i in range(7):
            x0 = gap + i * (size + gap)
            on = self.pressed == i + 1
            self.canvas.create_rectangle(x0, y0, x0 + size, y0 + size, outline=FG, width=3,
                                         fill=FG if on else BG)
            self.canvas.create_text(x0 + size / 2, y0 + size / 2, text=str(i + 1),
                                    fill=BG if on else FG, font=("Helvetica", int(size / 2), "bold"))

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    rospy.init_node("display_node")
    Display().run()
