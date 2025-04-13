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

export const cameraSettingsState = $state({
    AwbMode: AwbModes.Auto,
    AfMode: AfModes.Auto,
    AfTrigger: AfTriggers.Start
});

