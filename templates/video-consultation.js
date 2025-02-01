// Control Buttons
const muteButton = document.getElementById('muteButton');
const unmuteButton = document.getElementById('unmuteButton');
const endCallButton = document.getElementById('endCallButton');

// Chat Input
const chatInput = document.getElementById('chatInput');
const sendButton = document.getElementById('sendButton');
const chatMessages = document.getElementById('chatMessages');

// Mute Button Action
muteButton.addEventListener('click', () => {
  alert('Microphone muted.');
  // Logic to mute the microphone
});

// Unmute Button Action
unmuteButton.addEventListener('click', () => {
  alert('Microphone unmuted.');
  // Logic to unmute the microphone
});

// End Call Button Action
endCallButton.addEventListener('click', () => {
  alert('Ending the call...');
  // Logic to end the call
  window.location.href = "Dashboard.html"; // Redirect to dashboard
});

// Chat Functionality
sendButton.addEventListener('click', () => {
  const message = chatInput.value.trim();
  if (message) {
    const messageElement = document.createElement('div');
    messageElement.textContent = `You: ${message}`;
    chatMessages.appendChild(messageElement);
    chatInput.value = '';
    chatMessages.scrollTop = chatMessages.scrollHeight; // Scroll to the latest message
  } else {
    alert('Please enter a message!');
  }
});