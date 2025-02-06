// Use AgoraRTC from the global window object
const AgoraRTC = window.AgoraRTC;

// Agora Configuration (Dynamic Setup)
let APP_ID = "2db00ef90f14465e87e475b5bd14ba0b";
let TOKEN = "";
let CHANNEL_NAME = "demo";
let UID = 0;

const fetchAgoraConfig = async () => {
    try {
        
const response = await fetch(
    `https://agora.gleeze.com/token?channelName=${CHANNEL_NAME}`)
; // Adjust API endpoint
        const data = await response.json();
        //APP_ID = data.appId;
        TOKEN = data.token;
        UID = data.uid;
        console.log(TOKEN);
        //CHANNEL_NAME = data.channel;
    } catch (error) {
        console.error("Failed to fetch Agora config:", error);
    }
};

let client = AgoraRTC.createClient({ mode: "rtc", codec: "vp8" });
let localTrack, localAudioTrack;
let remoteUsers = {};

// HTML Elements
const startCallButton = document.getElementById("startCallButton");
const leaveCallButton = document.getElementById("leaveCallButton");
const muteButton = document.getElementById("muteButton");
const unmuteButton = document.getElementById("unmuteButton");

// Initialize media devices
navigator.mediaDevices.getUserMedia({ video: true, audio: true })
    .then(() => console.log("Camera & Microphone Access Granted"))
    .catch(err => console.error("Permission Denied:", err));

// Start Call
startCallButton.addEventListener("click", async () => {
    console.log("Attempting to join channel...");
    startCallButton.disabled = true;
    leaveCallButton.disabled = false;
    muteButton.disabled = false;
    unmuteButton.disabled = false;

    await fetchAgoraConfig();  // Fetch dynamic credentials before joining

    try {
        await client.join(APP_ID, CHANNEL_NAME, TOKEN, UID);
        console.log("Successfully joined channel:", CHANNEL_NAME);

        localTrack = await AgoraRTC.createCameraVideoTrack();
        localAudioTrack = await AgoraRTC.createMicrophoneAudioTrack();

        const localContainer = document.createElement("div");
        localContainer.id = "local-user";
        document.getElementById("local-video").appendChild(localContainer);
        localTrack.play(localContainer);

        await client.publish([localTrack, localAudioTrack]);
        console.log("Call started successfully!");
    } catch (error) {
        console.error("Error joining the call:", error);
    }
});

// Handle Remote Users Joining
client.on("user-published", async (user, mediaType) => {
    console.log("New user joined:", user.uid);
    await client.subscribe(user, mediaType);

    if (mediaType === "video") {
        const remoteContainer = document.createElement("div");
        remoteContainer.id = `remote-user-${user.uid}`;
        document.getElementById("remote-video").appendChild(remoteContainer);
        user.videoTrack.play(remoteContainer);
    }

    remoteUsers[user.uid] = user;
});

// Handle Remote Users Leaving
client.on("user-unpublished", (user) => {
    console.log("User left:", user.uid);
    const remoteContainer = document.getElementById(`remote-user-${user.uid}`);
    if (remoteContainer) remoteContainer.remove();
    delete remoteUsers[user.uid];
});

// Mute Button
muteButton.addEventListener("click", async () => {
    if (localAudioTrack) {
        await localAudioTrack.setMuted(true);
        console.log("Microphone muted.");
    }
});

// Unmute Button
unmuteButton.addEventListener("click", async () => {
    if (localAudioTrack) {
        await localAudioTrack.setMuted(false);
        console.log("Microphone unmuted.");
    }
});

// Leave Call
leaveCallButton.addEventListener("click", async () => {
    console.log("Leaving call...");
    leaveCallButton.disabled = true;
    startCallButton.disabled = false;
    muteButton.disabled = true;
    unmuteButton.disabled = true;

    await client.leave();
    if (localTrack) {
        localTrack.stop();
        localTrack.close();
    }
    if (localAudioTrack) {
        localAudioTrack.stop();
        localAudioTrack.close();
    }

    document.getElementById("local-video").innerHTML = "";
    document.getElementById("remote-video").innerHTML = "";

    console.log("Left the call.");
});
