#Requires AutoHotkey v2.0
#SingleInstance Force

; enable click animation ;;;;;;;;;;;


DllCall("SystemParametersInfo", "UInt", 0x101D, "UInt", 0, "UInt", 1, "UInt", 0) ;SPI_SETMOUSESONAR ON

OnExit(ExitSub)
ExitSub(*) {
    DllCall("SystemParametersInfo", "UInt", 0x101D, "UInt", 0, "UInt", 0, "UInt", 0) ;SPI_SETMOUSESONAR OFF
}


~LButton::{
    Send("{Ctrl down}")
    Send("{Ctrl up}")
}

~LButton Up::{
    Send("{Ctrl down}")
    Send("{Ctrl up}")
}
;;;;;;;;;;;



ToolTip("Waiting for the Hermes installer window to appear...")
winTitle := "Hermes"
try {
    WinWait(winTitle, , 30)
} catch {
    FileAppend("ERROR: Hermes installer window did not appear within 30s`n", "ahk.log")
    ExitApp(1)
}
ToolTip("Hermes window appeared. Sleeping for a second.....")

Sleep(5000)

WinGetPos(&x, &y, &w, &h, winTitle)
FileAppend(Format("Window found at x={1} y={2} w={3} h={4}`n", x, y, w, h), "ahk.log")
ToolTip("Clicking install")

; click install
clickX := x + (w / 2)
clickY := y + 418
Click(clickX, clickY)

Sleep(2000)
ToolTip("Done")

; done
ExitApp(0)