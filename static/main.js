document.addEventListener("DOMContentLoaded", () => {
  const chatBox = document.getElementById("chat-box");
  const userInput = document.getElementById("user-input");
  const sendBtn = document.getElementById("send-btn");

  function scrollToBottom() {
    chatBox.scrollTop = chatBox.scrollHeight;
  }

  // Add message with avatar
  function addMessage(msg, sender) {
    const wrapper = document.createElement("div");
    wrapper.className = `msg-wrapper ${sender}`;

    const avatar = document.createElement("div");
    avatar.className = "avatar";
    avatar.style.backgroundImage =
      sender === "user"
        ? "url('https://cdn-icons-png.flaticon.com/512/149/149071.png')" // user icon
        : "url('https://cdn-icons-png.flaticon.com/512/4712/4712038.png')"; // bot icon

    const msgDiv = document.createElement("div");
    msgDiv.className = sender === "user" ? "user-msg" : "bot-msg";
    msgDiv.textContent = msg;

    if (sender === "user") {
      wrapper.appendChild(msgDiv);
      wrapper.appendChild(avatar);
    } else {
      wrapper.appendChild(avatar);
      wrapper.appendChild(msgDiv);
    }

    chatBox.appendChild(wrapper);
    scrollToBottom();
  }

  // Typing animation
  function showTyping() {
    const wrapper = document.createElement("div");
    wrapper.className = "msg-wrapper bot";

    const avatar = document.createElement("div");
    avatar.className = "avatar";
    avatar.style.backgroundImage =
      "url('https://cdn-icons-png.flaticon.com/512/4712/4712038.png')";

    const typingDiv = document.createElement("div");
    typingDiv.className = "bot-msg typing";
    typingDiv.textContent = "Bot is typing...";

    wrapper.appendChild(avatar);
    wrapper.appendChild(typingDiv);

    chatBox.appendChild(wrapper);
    scrollToBottom();
    return wrapper;
  }

  async function sendMessage() {
    const msg = userInput.value.trim();
    if (!msg) return;

    addMessage(msg, "user");
    userInput.value = "";

    const typingWrapper = showTyping();

    try {
      const res = await fetch("/get", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ msg }),
      });

      const data = await res.json();
      typingWrapper.remove();
      addMessage(data.reply, "bot");
    } catch (err) {
      typingWrapper.remove();
      addMessage("⚠️ Server error. Try again.", "bot");
    }
  }

  sendBtn.addEventListener("click", sendMessage);
  userInput.addEventListener("keypress", (e) => {
    if (e.key === "Enter") sendMessage();
  });

  // Auto welcome message
  setTimeout(() => {
    addMessage("Hello 👋 I'm your College Assistant. How can I help you today?", "bot");
  }, 500);
});
