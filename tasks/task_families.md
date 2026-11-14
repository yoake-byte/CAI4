# The three task families

Your agent must handle three kinds of spoken request. Peers test it on all three (see the README), and the demo video shows all three. This file says what each family asks for, what counts as success, and gives example dialogues.

The dialogues are examples to build on, not the test set. Peers write their own, and the ones staff record to measure latency are not released.

Everything about the made-up venue comes from `toy_kb/knowledge_base.json`. Your agent may read that file however it likes. It is small enough that you should decide deliberately how much of it goes to the LLM on each turn, and then look at what that choice does to your waterfall.

## What every family shares

- The agent speaks in short sentences, because the user is listening and not reading. One or two sentences per reply is a good target.
- The agent never invents a fact. If the knowledge base does not hold the answer, it says so.
- The agent repeats back the values it acted on (the duration, the room, the time), so the user can catch a misheard number.
- When a request is missing something it needs, the agent asks one short question for it.
- The agent keeps what it needs from earlier turns. A dialogue is a conversation, not a list of unrelated commands.

## Family 1: timers and reminders

The user sets, checks, changes and cancels timers, and asks for reminders at a time of day. The agent keeps this state itself; the knowledge base is not involved.

What counts as success:

- It sets a timer for the length the user said, and says the length back.
- It answers "how much is left" with a plausible remaining time, counted from when the timer was set.
- It can extend, cancel and list timers and reminders, and handles two timers at once.
- A reminder carries the time of day and the thing to be reminded of. The agent says both back.
- If the user gives no length or no time, the agent asks for it.

Example 1:

```
User:  Start a ten minute timer for the tea.
Agent: Ten minutes for the tea, starting now.
User:  Actually make it fifteen.
Agent: Changed. Fifteen minutes for the tea.
User:  How long is left?
Agent: About fourteen minutes.
```

Example 2:

```
User:  Remind me to water the plants.
Agent: At what time?
User:  Six in the evening.
Agent: I will remind you to water the plants at six in the evening.
User:  What reminders do I have?
Agent: One reminder: water the plants, at six in the evening.
```

## Family 2: lookup

The user asks a factual question about the venue: opening hours, where something is, what a service costs, what a rule says.

What counts as success:

- The answer is right according to the knowledge base, and short.
- A question with two parts gets both parts answered.
- A question the knowledge base cannot answer gets an honest "I don't have that", not a guess. It may offer something related that it does know.

Example 1:

```
User:  What time does the cafe close on Saturdays?
Agent: Corner Cup closes at two in the afternoon on Saturdays.
```

Example 2:

```
User:  Where can I print, and how much is it?
Agent: The printing station is on the second floor. Black and white pages are ten cents, color pages fifty cents.
```

Example 3, the answer is not in the knowledge base:

```
User:  Is there a swimming pool?
Agent: I don't have anything about a pool. I can tell you about the cafe, the quiet lounge, the makerspace and the rooms.
```

## Family 3: booking a room with two constraints

The user books a meeting room. The request has two constraints that must both hold, such as the number of people and a piece of equipment, plus a day and time. They usually come out over several turns, and the user may change one halfway.

What counts as success:

- The agent collects the constraints it needs: the day and time, the number of people, and the equipment or feature.
- It proposes a room that satisfies all of them and is free at that time, and it names the room, the floor, the day and the time.
- It asks for a yes before booking, and ends with a confirmation line that repeats the room, the day, the time and the name.
- If no room satisfies both constraints, it says which constraint cannot be met together with which, and offers the closest alternative. It never books a room that breaks a constraint.
- If the user changes a constraint after a proposal, the agent searches again and does not keep the old room.

Example 1, a match:

```
User:  I need a room on Thursday afternoon.
Agent: How many people?
User:  Five.
Agent: Do you need any equipment?
User:  A projector.
Agent: Cedar Room on the first floor is free on Thursday at two in the afternoon. It holds eight and has a projector. Shall I book it?
User:  Yes, under the name Sam.
Agent: Booked. Cedar Room, Thursday at two in the afternoon, for Sam.
```

Example 2, no room satisfies both constraints:

```
User:  Can I get a room for fifteen people with a whiteboard?
Agent: No room holds fifteen and has a whiteboard. Maple Hall holds twenty but has no whiteboard. Would you like Maple Hall, or a smaller group?
User:  Maple Hall is fine. Wednesday at two.
Agent: Maple Hall is free on Wednesday at two in the afternoon. Shall I book it?
```

Example 3, a change of mind:

```
User:  A quiet room for two people on Monday at noon.
Agent: Oak Nook on the third floor is free on Monday at noon. It holds two and is quiet. Shall I book it?
User:  Make it three people instead.
Agent: Then no quiet room is free on Monday at noon. Pine Room holds four and is quiet, but it is free only on Tuesday, Thursday and Friday. Would you like one of those days?
```

## Writing your own test dialogues

Write at least ten of your own, three or four per family, before you record anything. Include one dialogue per family where the agent cannot give the user what they ask for: a missing value, an unknown fact, a booking that cannot be satisfied. Agents are judged on how they fail as much as on how they succeed.
