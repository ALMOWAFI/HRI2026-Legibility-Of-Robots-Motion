#!/usr/bin/env python3
"""Display node: fullscreen window on the participant's monitor (Tkinter), plus the keypad.

Sub:  /display       (String, JSON from experiment_node)
        {"screen": "text" | "reach" | "score" | "rating", "title", "text",
         "cubes": [colour, ...] left to right as the participant sees them, "choice": index or -1}
Pub:  /rating        (std_msgs/Int32)  keys 1-7 (number row or keypad) while the window has focus
      /button_press  (std_msgs/Int32)  keys a s d f -> 0-3, only with display.keyboard_buttons
                                       (dry runs without the push buttons)
Esc leaves fullscreen.
"""
import json
import queue
import tkinter as tk

import rospy
from std_msgs.msg import Int32, String

COLORS = {"orange": "#ff8000", "red": "#e00000", "yellow": "#ffd800", "green": "#00b000",
          "blue": "#1060ff", "purple": "#9020c0", "white": "#ffffff"}
BG, FG, DIM = "#101010", "#f0f0f0", "#808080"
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
        self.root.attributes("-fullscreen", bool(cfg.get("fullscreen", True)))
        self.root.bind("<Escape>", lambda e: self.root.attributes("-fullscreen", False))
        self.root.bind("<Key>", self.on_key)
        self.canvas = tk.Canvas(self.root, bg=BG, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda e: self.draw())
        self.state = {"screen": "text", "title": "Please wait", "text": "", "cubes": [], "choice": -1}
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
            self.state.update(json.loads(self.messages.get_nowait()))
            self.pressed = None
            changed = True
        if changed:
            self.draw()
        self.root.after(50, self.poll)

    def draw(self):
        c, s = self.canvas, self.state
        c.delete("all")
        w, h = c.winfo_width(), c.winfo_height()
        big = max(24, h // 12)
        c.create_text(w / 2, h * 0.15, text=s.get("title", ""), fill=FG, font=("Helvetica", big, "bold"))
        if s["screen"] in ("reach", "score") and s.get("cubes"):
            self.draw_cubes(w, h, s["cubes"], s.get("choice", -1))
        elif s["screen"] == "rating":
            self.draw_scale(w, h)
        if s.get("text"):
            c.create_text(w / 2, h * 0.82, text=s["text"], fill=FG, width=w * 0.8,
                          font=("Helvetica", max(16, h // 28)), justify="center")

    def draw_cubes(self, w, h, cubes, choice):
        n = len(cubes)
        size = min(w / (n * 1.6), h * 0.3)
        gap = (w - n * size) / (n + 1)
        y0 = h * 0.38
        for i, color in enumerate(cubes):
            x0 = gap + i * (size + gap)
            chosen = i == choice
            self.canvas.create_rectangle(x0, y0, x0 + size, y0 + size, fill=COLORS.get(color, color),
                                         outline=FG if chosen else BG, width=12 if chosen else 1,
                                         stipple="" if chosen or choice < 0 else "gray50")
            if chosen:
                self.canvas.create_text(x0 + size / 2, y0 + size + h * 0.05, text="YOUR CHOICE",
                                        fill=FG, font=("Helvetica", max(14, h // 30), "bold"))

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
