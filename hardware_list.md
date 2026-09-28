# Hardware Shopping List

The parts for the Arduino alert system: RGB LED, vibration motor, buzzer and acknowledgement button. The **laptop's built-in webcam** is the camera, so no external webcam is needed.

Estimated total: **₹700–1000**. Prices are approximate (India, online).

Each item below has the exact spec, what to search for, what **not** to buy, and how to check the part when it arrives.

---

## Quick checklist

| # | Part | Qty (incl. spares) | ≈ Price |
|---|---|---|---|
| 1 | Arduino Uno R3 (clone OK) | 1 | ₹400–600 |
| 2 | USB A-to-B cable | 1 | ₹50–100 (often included with #1) |
| 3 | 5 mm RGB LED, **common-cathode**, diffused | 3 | ₹15–30 |
| 4 | 220 Ω resistor, 1/4 W | 3 (pack of 10+) | ₹10–20 |
| 5 | 10 mm coin vibration motor, 3 V | 2 | ₹60–100 |
| 6 | 2N2222 / PN2222A NPN transistor, TO-92 | 3 | ₹10–15 |
| 7 | 1N4007 diode | 2 | ₹5–10 |
| 8 | 1 kΩ resistor, 1/4 W | 1 (pack of 10+) | ₹10–20 |
| 9 | 5 V **active** piezo buzzer | 1 | ₹15–30 |
| 10 | 6×6 mm tactile push button | 2 | ₹5–10 |
| 11 | 400-point breadboard | 1 | ₹60–100 |
| 12 | Male-to-male jumper wires, ~20 cm | 1 pack (20–40) | ₹60–100 |

---

## Item details

### 1. Arduino Uno R3 (microcontroller board)
- **What it does:** receives risk levels from the laptop over USB serial (`R:<level>\n`) and drives the LED, motor and buzzer. It also sends `ACK\n` when the button is pressed.
- **Search for:** "Arduino Uno R3 compatible board CH340" (a clone is fine) or "Arduino Uno R3 original".
- **Spec:** ATmega328P chip, 5 V logic, USB-B port. Both the long DIP chip and the small square SMD chip versions work.
- **Don't buy:** Arduino **Nano**, **Mega**, **Leonardo**, **Uno R4**, ESP32 or NodeMCU. They would work with changes, but our code and wiring assume a Uno R3.
- **On arrival:** plug it into the laptop. It should appear under *Ports (COM & LPT)* in Device Manager. If it's a CH340 clone and doesn't appear, install the **CH340 driver**.

### 2. USB A-to-B cable ("printer cable")
- **What it does:** serial link and power between the laptop and the Uno. No separate power supply is needed.
- **Search for:** "USB A to B cable Arduino".
- **Check first:** most Uno kits include it. If your laptop has **only USB-C** ports, buy a **USB-C to USB-B** cable instead.
- **Don't buy:** micro-USB or mini-USB cables. Those are for the Nano.

### 3. RGB LED: 5 mm, common-cathode, diffused
- **What it does:** shows the risk level: green = safe, yellow = drowsiness predicted soon, red = critical. It also shows a fault blink if the laptop stops sending messages for 2 s.
- **Search for:** "5mm RGB LED common cathode diffused 4 pin".
- **Spec:** 4 legs. The **longest leg is the common one and goes to GND**.
- **Don't buy:** **common-anode** RGB LEDs. They look identical, but the wiring and code are inverted. Avoid single-colour LEDs and "addressable" LEDs (WS2812/NeoPixel). **Diffused** (milky) is better than clear because the colours blend into a proper yellow.
- **Tip:** if a listing doesn't say "cathode" or "anode", don't buy it.

### 4. 220 Ω resistors (for the LED)
- **What it does:** one in series with each LED colour, so the LED and the Arduino pin don't burn out.
- **Search for:** "220 ohm resistor 1/4 watt".
- **Colour bands:** red, red, brown, gold (4-band), or red, red, black, black, brown (5-band).
- **Don't buy:** SMD resistors (tiny rectangles that won't fit a breadboard). Buy through-hole ones with wire legs.

### 5. Coin vibration motor: 10 mm, 3 V
- **What it does:** the level-2 alert, a haptic buzz you feel when drowsiness is predicted.
- **Search for:** "coin vibration motor 10mm 3V 1027" (or "1034").
- **Spec:** flat disc about 10 mm across and 2.5–3.5 mm thick, with two thin wire leads (red +, blue or black −). Rated about 3 V and 60–90 mA.
- **Don't buy:** "vibration motor **module**" boards (a small PCB with the motor and a transistor already on it). They hide the transistor circuit the project should demonstrate. Also avoid large cylindrical "ERM" pager motors and motors rated 12 V.
- **Note:** the wire leads are thin and floppy. Twist them around a header pin or solder them so they sit firmly in the breadboard. The code will run the motor at about 60% power via PWM, because the Uno supplies 5 V and the motor is rated 3 V.

### 6. NPN transistor: 2N2222 or PN2222A (TO-92)
- **What it does:** a switch that lets a weak Arduino pin turn on the motor. The motor draws more current than a pin can supply safely, so it must **never** connect directly to a pin.
- **Search for:** "2N2222A transistor TO-92" or "PN2222A".
- **Spec:** small black plastic half-cylinder with 3 legs (the TO-92 package). The legs are Emitter, Base and Collector. **Check the datasheet for your exact part number**, because the leg order varies between manufacturers.
- **Don't buy:** the metal-can version (TO-18). It works but is awkward on a breadboard. Don't substitute a **PNP** transistor (e.g. 2N2907), which needs different wiring. BC547 also works as a substitute if 2N2222 is unavailable.

### 7. 1N4007 diode (flyback diode)
- **What it does:** placed across the motor, it absorbs the voltage spike the motor produces when it switches off. Without it, that spike can kill the transistor over time.
- **Search for:** "1N4007 diode".
- **Spec:** small black cylinder with a **silver/grey stripe**. The stripe is the cathode and faces **+5 V**.
- **Don't buy:** LEDs or Zener diodes. **1N4148** (small glass body, black stripe) is also fine if that's what you have.

### 8. 1 kΩ resistor (transistor base)
- **What it does:** limits the current from the Arduino pin into the transistor's base.
- **Search for:** "1k ohm resistor 1/4 watt".
- **Colour bands:** brown, black, red, gold (4-band), or brown, black, black, brown, brown (5-band).
- **Tip:** a mixed resistor kit (10 Ω–1 MΩ, about ₹100–150) covers items 4 and 8 and more.

### 9. Active piezo buzzer: 5 V
- **What it does:** the level-3 alert, a loud beep when you're drowsy *now*.
- **Search for:** "5V active buzzer".
- **Spec:** black cylinder about 12 mm across. The **bottom is sealed with black epoxy**, and there's often a white sticker on top. The longer leg is +.
- **Don't buy:** a **passive** buzzer. It looks almost identical, but the bottom shows an **exposed green circuit board**, and it needs tone-generation code to make any sound.
- **On arrival:** connect it directly to the Uno's 5 V and GND pins. An active buzzer beeps continuously. A passive one stays silent or only clicks.

### 10. Tactile push button: 6×6 mm
- **What it does:** the driver's "I'm awake" button. It resets the alert and is logged as an acknowledgement.
- **Search for:** "6x6mm tactile push button 4 pin".
- **Spec:** small square button with 4 legs. The legs are internally connected in pairs, so place it **across the centre gap** of the breadboard. No resistor is needed because the Arduino's built-in pull-up is used.
- **Don't buy:** latching buttons (the ones that stay pressed) or 2-pin panel-mount switches. Either works, but the 6×6 tactile type fits the breadboard directly.

### 11. Breadboard: 400-point (half size)
- **What it does:** lets you build the circuit without soldering.
- **Search for:** "400 point breadboard" (or 830-point if you want more room).
- **Don't buy:** "mini" 170-point boards. They're too small for this circuit.

### 12. Jumper wires: male-to-male, ~20 cm
- **What it does:** connects the Arduino headers to the breadboard.
- **Search for:** "male to male jumper wires 20cm 40 pcs".
- **Don't buy:** only female-to-female or male-to-female sets. Those are for modules with pin headers, not breadboards.

---

## Buying a starter kit instead

An **"Arduino Uno R3 starter kit"** (about ₹800–1200) usually covers most of the list. Before buying, check that the listing includes:

- [ ] Uno R3 board and USB cable
- [ ] RGB LED, **common-cathode**
- [ ] 220 Ω and 1 kΩ resistors
- [ ] 2N2222 (or BC547) NPN transistor
- [ ] 1N4007 diode
- [ ] **Active** buzzer (many kits include both an active and a passive one)
- [ ] Push buttons, breadboard, jumper wires

Kits almost never include the **coin vibration motor (#5)**, so buy it separately.

---

## Not needed

- External webcam (the laptop's built-in camera is used)
- External power supply or batteries (USB power from the laptop is enough)
- Soldering iron (optional, only to tidy the motor leads)
- Pull-up resistor for the button (the Arduino's built-in one is used)
- Multimeter (optional but handy for debugging)
