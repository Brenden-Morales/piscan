export enum AwbModes {
    "Auto" = 0,
    "Tungsten",
    "Fluorescent",
    "Indoor",
    "Daylight",
    "Cloudy",
    "Custom"
}

export enum AfModes {
    "Manual" = 0,
    "Auto",
    "Continuous"
}

export enum AfTriggers {
    "Start" = 0,
    "Cancel"
}

export enum AfRange {
    "Normal" = 0,
    "Macro",
    "Full"
}

export enum AfSpeed {
    "Normal" = 0,
    "Fast"
}

export enum PiCams {
    "picam0" = 0,
    "picam1",
    "picam2",
    "picam3",
    "picam4",
    "picam5"
}

export const cameraSettingsState = $state({
    AwbMode: AwbModes.Auto,
    AfMode: AfModes.Auto,
    AfTrigger: AfTriggers.Start,
    AfRange: AfRange.Normal,
    AfSpeed: AfSpeed.Normal,
    AnalogueGain: 1,
    ExposureTime: 20000,
    SelectedCamera: 0,
    Timestamp: Date.now(),
    Loading: false
});

