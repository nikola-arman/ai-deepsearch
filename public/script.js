document.addEventListener('DOMContentLoaded', () => {
            // DOM elements
            const chatMessages = document.getElementById('chat-messages');
            const userInput = document.getElementById('user-input');
            const sendBtn = document.getElementById('send-btn');
            const newChatBtn = document.getElementById('new-chat-btn');
            const chatHistory = document.getElementById('chat-history');

            // Chat state
            let currentChatId = null;
            let chats = JSON.parse(localStorage.getItem('chats') || '{}');
            let activeResearchTaskId = localStorage.getItem('activeResearchTaskId');
            let pollingInterval = null;

            // Initialize app
            function init() {
                loadChatHistory();

                if (Object.keys(chats).length === 0) {
                    createNewChat();
                } else {
                    const lastChatId = localStorage.getItem('lastChatId');
                    if (lastChatId && chats[lastChatId]) {
                        loadChat(lastChatId);
                    } else {
                        const firstChatId = Object.keys(chats)[0];
                        loadChat(firstChatId);
                    }
                }

                // Check for active research task
                if (activeResearchTaskId) {
                    const lastMessage = getLastResearchMessage();
                    if (lastMessage && lastMessage.status === 'running') {
                        startPolling();
                    } else {
                        // Clear the active task if it's not running
                        localStorage.removeItem('activeResearchTaskId');
                        activeResearchTaskId = null;
                    }
                }

                // Initialize sidebar toggle
                const sidebarToggle = document.getElementById('sidebar-toggle');
                const sidebar = document.querySelector('.sidebar');
                const chatContainer = document.querySelector('.chat-container');

                // Check if sidebar state is saved
                const isSidebarCollapsed = localStorage.getItem('sidebarCollapsed') === 'true';
                if (isSidebarCollapsed) {
                    sidebar.classList.add('collapsed');
                    chatContainer.classList.add('expanded');
                }

                sidebarToggle.addEventListener('click', () => {
                    sidebar.classList.toggle('collapsed');
                    chatContainer.classList.toggle('expanded');
                    // Save sidebar state
                    localStorage.setItem('sidebarCollapsed', sidebar.classList.contains('collapsed'));
                });

                // Event listeners
                userInput.addEventListener('keydown', (e) => {
                    if (e.key === 'Enter' && !e.shiftKey) {
                        e.preventDefault();
                        sendMessage();
                    }
                });

                sendBtn.addEventListener('click', sendMessage);
                newChatBtn.addEventListener('click', createNewChat);
            }

            // Create a new chat
            function createNewChat() {
                const chatId = Date.now().toString();
                const chatTitle = "New Chat";

                chats[chatId] = {
                    title: chatTitle,
                    messages: []
                };

                saveChats();
                loadChatHistory();
                loadChat(chatId);
            }

            // Load chat history sidebar
            function loadChatHistory() {
                chatHistory.innerHTML = '';

                Object.entries(chats).forEach(([id, chat]) => {
                    const historyItem = document.createElement('div');
                    historyItem.className = 'history-item';
                    historyItem.dataset.chatId = id;

                    if (id === currentChatId) {
                        historyItem.classList.add('active');
                    }

                    // Create title span
                    const titleSpan = document.createElement('span');
                    titleSpan.className = 'history-item-title';
                    titleSpan.textContent = chat.title;
                    titleSpan.addEventListener('click', () => loadChat(id));

                    // Create actions container
                    const actionsDiv = document.createElement('div');
                    actionsDiv.className = 'history-item-actions';

                    // Create edit button
                    const editBtn = document.createElement('button');
                    editBtn.className = 'edit-chat-btn';
                    editBtn.innerHTML = '✏️'; // pencil symbol
                    editBtn.title = 'Rename Chat';

                    // Create delete button
                    const deleteBtn = document.createElement('button');
                    deleteBtn.className = 'delete-chat-btn';
                    deleteBtn.innerHTML = '&times;'; // × symbol
                    deleteBtn.title = 'Delete Chat';

                    // Add event listeners
                    editBtn.addEventListener('click', (e) => {
                        e.stopPropagation();
                        startEditingChatTitle(historyItem, id, chat.title);
                    });

                    deleteBtn.addEventListener('click', (e) => {
                        e.stopPropagation();
                        deleteChat(id);
                    });

                    // Add buttons to actions container
                    actionsDiv.appendChild(editBtn);
                    actionsDiv.appendChild(deleteBtn);

                    // Add elements to history item
                    historyItem.appendChild(titleSpan);
                    historyItem.appendChild(actionsDiv);

                    chatHistory.appendChild(historyItem);
                });
            }

            // Start editing a chat title
            function startEditingChatTitle(historyItem, chatId, currentTitle) {
                // Replace title with input field
                const titleSpan = historyItem.querySelector('.history-item-title');
                const actionsDiv = historyItem.querySelector('.history-item-actions');

                historyItem.removeChild(titleSpan);
                historyItem.removeChild(actionsDiv);

                // Create input and buttons
                const inputContainer = document.createElement('div');
                inputContainer.style.display = 'flex';
                inputContainer.style.width = '100%';

                const input = document.createElement('input');
                input.className = 'chat-title-input';
                input.type = 'text';
                input.value = currentTitle;

                const saveBtn = document.createElement('button');
                saveBtn.className = 'edit-chat-btn';
                saveBtn.innerHTML = '✓'; // checkmark
                saveBtn.title = 'Save';

                const cancelBtn = document.createElement('button');
                cancelBtn.className = 'delete-chat-btn';
                cancelBtn.innerHTML = '✕'; // x
                cancelBtn.title = 'Cancel';

                // Event listeners
                saveBtn.addEventListener('click', () => saveChatTitle(chatId, input.value));
                cancelBtn.addEventListener('click', () => loadChatHistory()); // Just reload to cancel

                input.addEventListener('keydown', (e) => {
                    if (e.key === 'Enter') {
                        saveChatTitle(chatId, input.value);
                    } else if (e.key === 'Escape') {
                        loadChatHistory();
                    }
                });

                // Add to container
                inputContainer.appendChild(input);
                inputContainer.appendChild(saveBtn);
                inputContainer.appendChild(cancelBtn);
                historyItem.appendChild(inputContainer);

                // Focus input
                input.focus();
                input.select();
            }

            // Save edited chat title
            function saveChatTitle(chatId, newTitle) {
                if (newTitle.trim() === '') {
                    // If empty, use a default title
                    newTitle = 'Chat ' + new Date().toLocaleDateString();
                }

                chats[chatId].title = newTitle.trim();
                saveChats();
                loadChatHistory();
            }

            // Delete a chat
            function deleteChat(chatId) {
                if (confirm('Are you sure you want to delete this chat?')) {
                    // Remove the chat
                    delete chats[chatId];
                    saveChats();

                    // If the current chat was deleted, load another chat or create a new one
                    if (chatId === currentChatId) {
                        if (Object.keys(chats).length > 0) {
                            loadChat(Object.keys(chats)[0]);
                        } else {
                            createNewChat();
                        }
                    }

                    loadChatHistory();
                }
            }

            // Load a specific chat
            function loadChat(chatId) {
                currentChatId = chatId;
                localStorage.setItem('lastChatId', chatId);

                // Update history sidebar
                document.querySelectorAll('.history-item').forEach(item => {
                    item.classList.toggle('active', item.dataset.chatId === chatId);
                });

                // Display chat messages
                displayMessages(chats[chatId].messages);
            }

            // Configure marked options
            marked.setOptions({
                breaks: true,
                gfm: true,
                headerIds: false,
                mangle: false
            });

            // Display messages in the chat
            function displayMessages(messages) {
                chatMessages.innerHTML = '';

                if (messages.length === 0) {
                    // Show welcome message if chat is empty
                    const welcomeDiv = document.createElement('div');
                    welcomeDiv.className = 'welcome-message';
                    welcomeDiv.innerHTML = '<h2>Medical AI Assistant</h2><p>How can I help you today?</p>';
                    chatMessages.appendChild(welcomeDiv);
                    return;
                }

                messages.forEach(msg => {
                    const messageContainer = document.createElement('div');
                    messageContainer.className = `message-container ${msg.role}-container`;

                    const messageDiv = document.createElement('div');
                    messageDiv.className = `message ${msg.role}-message`;

                    const avatarDiv = document.createElement('div');
                    avatarDiv.className = `message-avatar ${msg.role}-avatar`;
                    avatarDiv.textContent = msg.role === 'user' ? 'U' : 'AI';

                    const contentDiv = document.createElement('div');
                    contentDiv.className = 'message-content';

                    // Handle different message types
                    if (msg.type === 'research') {
                        // Research message with loading state
                        const researchDiv = document.createElement('div');
                        researchDiv.className = 'research-message';

                        const statusDiv = document.createElement('div');
                        statusDiv.className = 'research-status';
                        statusDiv.setAttribute('data-status', msg.status);

                        if (msg.status === 'running') {
                            const loadingCircle = document.createElement('div');
                            loadingCircle.className = 'loading-circle';
                            statusDiv.appendChild(loadingCircle);
                            statusDiv.appendChild(document.createTextNode('Running...'));
                        } else if (msg.status === 'failed') {
                            statusDiv.textContent = 'Research failed. Please try again.';
                        } else if (msg.status === 'completed') {
                            statusDiv.textContent = 'Research completed';
                        }

                        researchDiv.appendChild(statusDiv);

                        if (msg.content) {
                            const contentText = document.createElement('div');
                            contentText.className = 'markdown-content';
                            contentText.innerHTML = marked.parse(msg.content);
                            researchDiv.appendChild(contentText);
                        }

                        if (msg.sources && msg.sources.length > 0) {
                            const sourcesDiv = document.createElement('div');
                            sourcesDiv.className = 'sources';

                            const sourcesTitle = document.createElement('div');
                            sourcesTitle.className = 'sources-title';
                            sourcesTitle.textContent = 'References:';

                            const sourcesList = document.createElement('ul');
                            sourcesList.className = 'sources-list';

                            msg.sources.forEach(source => {
                                const listItem = document.createElement('li');
                                const link = document.createElement('a');
                                link.href = source.url;
                                link.target = '_blank';
                                link.textContent = source.title;
                                listItem.appendChild(link);
                                sourcesList.appendChild(listItem);
                            });

                            sourcesDiv.appendChild(sourcesTitle);
                            sourcesDiv.appendChild(sourcesList);
                            researchDiv.appendChild(sourcesDiv);
                        }

                        contentDiv.appendChild(researchDiv);
                    } else {
                        // Regular message
                        const contentText = document.createElement('div');
                        contentText.className = 'markdown-content';
                        contentText.innerHTML = marked.parse(msg.content);
                        contentDiv.appendChild(contentText);
                    }

                    // For user messages, append avatar after content for right alignment
                    if (msg.role === 'user') {
                        messageDiv.appendChild(contentDiv);
                        messageDiv.appendChild(avatarDiv);
                    } else {
                        messageDiv.appendChild(avatarDiv);
                        messageDiv.appendChild(contentDiv);
                    }

                    messageContainer.appendChild(messageDiv);
                    chatMessages.appendChild(messageContainer);
                });

                // Scroll to bottom
                chatMessages.scrollTop = chatMessages.scrollHeight;
            }

            // Format sources for display
            function formatSources(sources) {
                if (!sources || sources.length === 0) return '';
                return `
            <div class="sources-title">References:</div>
            <ul class="sources-list">
                ${sources.map(source => `
                    <li><a href="${source.url}" target="_blank">${source.title}</a></li>
                `).join('')}
            </ul>
        `;
    }

    // Send a message
    async function sendMessage() {
        const message = userInput.value.trim();
        if (!message) return; // Don't send empty messages
        
        // Add user message to chat
        addMessage('user', message);
        
        // Clear input
        userInput.value = '';
        
        // Start research task with full chat history
        await startResearchTask();
    }

    // Add a message to the chat
    function addMessage(role, content, type = 'normal', status = null, sources = null) {
        if (!currentChatId) return;
        
        const message = {
            role: role, // 'user' or 'assistant'
            content: content,
            type: type,
            status: status,
            sources: sources,
            timestamp: new Date().toISOString()
        };
        
        chats[currentChatId].messages.push(message);
        saveChats();
        displayMessages(chats[currentChatId].messages);
    }

    // Start a research task
    async function startResearchTask() {
        try {
            // Add initial research message
            addMessage('assistant', 'DeepSearch AI is running...', 'research', 'running');
            
            // Get all messages from current chat
            const currentMessages = chats[currentChatId].messages
                .filter(msg => msg.type === 'normal') // Only include normal messages
                .map(msg => ({
                    role: msg.role,
                    content: msg.content
                }));
            
            // Send request to start research with full chat history
            const response = await fetch('/v1/chat/completions', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    messages: currentMessages
                })
            });
            
            if (!response.ok) {
                const errorData = await response.json();
                if (response.status === 500 && errorData.detail) {
                    addMessage('assistant', errorData.detail, 'research', 'failed');
                } else {
                    throw new Error('Failed to start research');
                }
                return;
            }
            
            const data = await response.json();
            activeResearchTaskId = data.task_id;
            
            // Start polling for results
            startPolling();
            
        } catch (error) {
            console.error('Error starting research:', error);
            addMessage('assistant', 'Sorry, there was an error starting the research.', 'research', 'failed');
        }
    }

    // Get the last research message from the current chat
    function getLastResearchMessage() {
        if (!currentChatId || !chats[currentChatId]) return null;
        
        const messages = chats[currentChatId].messages;
        for (let i = messages.length - 1; i >= 0; i--) {
            if (messages[i].type === 'research') {
                return messages[i];
            }
        }
        return null;
    }

    // Start polling for research results
    function startPolling() {
        if (pollingInterval) {
            clearInterval(pollingInterval);
        }
        
        // Save the active task ID
        localStorage.setItem('activeResearchTaskId', activeResearchTaskId);
        
        pollingInterval = setInterval(async () => {
            try {
                const response = await fetch(`/v1/research/${activeResearchTaskId}`);
                if (!response.ok) {
                    throw new Error('Failed to check research status');
                }
                
                const data = await response.json();
                
                if (data.status === 'completed') {
                    // Research completed
                    clearInterval(pollingInterval);
                    pollingInterval = null;
                    localStorage.removeItem('activeResearchTaskId');
                    activeResearchTaskId = null;
                    
                    // Update the message with results
                    const lastMessage = chats[currentChatId].messages[chats[currentChatId].messages.length - 1];
                    lastMessage.content = data.results;
                    lastMessage.status = 'completed';
                    lastMessage.sources = data.results.sources;
                    lastMessage.type = 'normal'; // Switch to normal message type
                    
                    saveChats();
                    displayMessages(chats[currentChatId].messages);
                    
                } else if (data.status === 'failed') {
                    // Research failed
                    clearInterval(pollingInterval);
                    pollingInterval = null;
                    localStorage.removeItem('activeResearchTaskId');
                    activeResearchTaskId = null;
                    
                    const lastMessage = chats[currentChatId].messages[chats[currentChatId].messages.length - 1];
                    lastMessage.content = data.error || 'Sorry, there was an error during research.';
                    lastMessage.status = 'failed';
                    lastMessage.type = 'normal'; // Switch to normal message type
                    
                    saveChats();
                    displayMessages(chats[currentChatId].messages);
                }
                
            } catch (error) {
                console.error('Error polling research status:', error);
                clearInterval(pollingInterval);
                pollingInterval = null;
                localStorage.removeItem('activeResearchTaskId');
                activeResearchTaskId = null;
                
                const lastMessage = chats[currentChatId].messages[chats[currentChatId].messages.length - 1];
                lastMessage.content = 'Sorry, there was an error checking the research status.';
                lastMessage.status = 'failed';
                lastMessage.type = 'normal'; // Switch to normal message type
                
                saveChats();
                displayMessages(chats[currentChatId].messages);
            }
        }, 2000); // Poll every 2 seconds
    }

    // Save chats to localStorage
    function saveChats() {
        localStorage.setItem('chats', JSON.stringify(chats));
    }

    // Initialize the app
    init();
});