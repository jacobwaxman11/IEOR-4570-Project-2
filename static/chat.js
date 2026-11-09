const PROMPTS = [
    "1-bedroom apartments in Chelsea under $6,000",
    "What's the cheapest studio in NoMad?",
    "Best deals in Hell's Kitchen",
    "Compare rents in Chelsea and the West Village",
    "What amenities does 95 Wall Street have?",
];

const SORTS = [
    ["recent", "Recently saved"],
    ["rent_asc", "Rent, low to high"],
    ["rent_desc", "Rent, high to low"],
    ["neighborhood", "Neighborhood"],
    ["beds", "Bedrooms"],
    ["deal", "Deal score"],
];

const HEART = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 20s-7-4.4-7-9.1C5 8 6.8 6.2 9 6.2c1.3 0 2.4.6 3 1.6.6-1 1.7-1.6 3-1.6 2.2 0 4 1.8 4 4.7C19 15.6 12 20 12 20z"></path></svg>';

const STORE = "rental-analyst-session";

const rail = document.getElementById("rail");
const railBody = document.getElementById("rail-body");
const scrim = document.getElementById("scrim");
const log = document.getElementById("log");
const logScroll = document.getElementById("log-scroll");
const form = document.getElementById("form");
const input = document.getElementById("input");
const sendBtn = document.getElementById("send");
const topTitle = document.getElementById("top-title");
const tabChats = document.getElementById("tab-chats");
const tabSaved = document.getElementById("tab-saved");

const state = {
    sessionId: null,
    title: "New chat",
    tab: "chats",
    sort: "recent",
    threads: [],
    favorites: [],
    sending: false,
};

function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text != null) node.textContent = text;
    return node;
}

function escapeHtml(value) {
    return String(value).replace(/[&<>"']/g, (char) => ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#39;",
    }[char]));
}

function inline(text) {
    let value = escapeHtml(text);
    value = value.replace(/`([^`]+)`/g, "<code>$1</code>");
    value = value.replace(
        /\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g,
        '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>',
    );
    value = value.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
    value = value.replace(/(^|[\s(])\*([^*]+)\*(?=[\s).,]|$)/g, "$1<em>$2</em>");
    return value;
}

function renderTable(lines) {
    const cells = (line) => line.trim().replace(/^\||\|$/g, "").split("|").map((cell) => inline(cell.trim()));
    const header = cells(lines[0]);
    const body = lines.slice(1).filter((line) => !/^[\s|:-]+$/.test(line));
    const head = `<tr>${header.map((cell) => `<th>${cell}</th>`).join("")}</tr>`;
    const rows = body.map((line) => `<tr>${cells(line).map((cell) => `<td>${cell}</td>`).join("")}</tr>`).join("");
    return `<table><thead>${head}</thead><tbody>${rows}</tbody></table>`;
}

function indentOf(line) {
    return line.match(/^[ \t]*/)[0].replace(/\t/g, "    ").length;
}

function listMatch(line) {
    const match = line.match(/^([ \t]*)([-*]|\d+\.)[ \t]+(.*)$/);
    if (!match) return null;
    return {
        indent: match[1].replace(/\t/g, "    ").length,
        ordered: /\d/.test(match[2]),
        text: match[3],
    };
}

function dedentLines(lines) {
    const widths = lines.filter((line) => line.trim()).map(indentOf);
    const cut = widths.length ? Math.min(...widths) : 0;
    return lines.map((line) => line.slice(cut)).join("\n").trim();
}

function readItem(lines, index) {
    const first = listMatch(lines[index]);
    const nested = [];
    let cursor = index + 1;
    while (cursor < lines.length) {
        const line = lines[cursor];
        if (!line.trim()) {
            let look = cursor + 1;
            while (look < lines.length && !lines[look].trim()) look += 1;
            if (look >= lines.length) break;
            const next = listMatch(lines[look]);
            if (next && next.indent === first.indent && next.ordered === first.ordered) break;
            if (indentOf(lines[look]) <= first.indent) break;
            nested.push(line);
            cursor += 1;
            continue;
        }
        const next = listMatch(line);
        if (next && next.indent === first.indent && next.ordered === first.ordered) break;
        if (indentOf(line) <= first.indent) break;
        nested.push(line);
        cursor += 1;
    }
    return { first, nested, next: cursor };
}

function readList(lines, index) {
    const opener = listMatch(lines[index]);
    const items = [];
    let cursor = index;
    while (cursor < lines.length) {
        if (!lines[cursor].trim()) {
            let look = cursor + 1;
            while (look < lines.length && !lines[look].trim()) look += 1;
            const next = look < lines.length ? listMatch(lines[look]) : null;
            if (!next || next.indent !== opener.indent || next.ordered !== opener.ordered) break;
            cursor = look;
        }
        const item = listMatch(lines[cursor]);
        if (!item || item.indent !== opener.indent || item.ordered !== opener.ordered) break;
        const read = readItem(lines, cursor);
        items.push(read);
        cursor = read.next;
    }
    return { ordered: opener.ordered, items, next: cursor };
}

function renderMarkdown(source) {
    const fences = [];
    const text = String(source || "").replace(/```[^\n]*\n([\s\S]*?)```/g, (_, code) => {
        fences.push(`<pre class="md-code"><code>${escapeHtml(code.replace(/\n$/, ""))}</code></pre>`);
        return `\u0000F${fences.length - 1}\u0000`;
    });
    const lines = text.split("\n");
    const html = [];
    const paragraph = [];

    function flushParagraph() {
        if (!paragraph.length) return;
        html.push(`<p>${paragraph.map((line) => inline(line)).join("<br>")}</p>`);
        paragraph.length = 0;
    }

    let index = 0;
    while (index < lines.length) {
        const line = lines[index];
        const fence = line.trim().match(/^\u0000F(\d+)\u0000$/);
        if (fence) {
            flushParagraph();
            html.push(fences[Number(fence[1])]);
            index += 1;
            continue;
        }
        if (!line.trim()) {
            flushParagraph();
            index += 1;
            continue;
        }
        const heading = line.match(/^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$/);
        if (heading) {
            flushParagraph();
            html.push(`<p class="md-h md-h-${heading[1].length}">${inline(heading[2])}</p>`);
            index += 1;
            continue;
        }
        if (listMatch(line)) {
            flushParagraph();
            const list = readList(lines, index);
            index = list.next;
            const tag = list.ordered ? "ol" : "ul";
            const items = list.items.map((item) => {
                const nested = item.nested.length ? renderMarkdown(dedentLines(item.nested)) : "";
                return `<li>${inline(item.first.text)}${nested}</li>`;
            });
            html.push(`<${tag}>${items.join("")}</${tag}>`);
            continue;
        }
        if (/^\s*\|.+\|\s*$/.test(line)) {
            flushParagraph();
            const rows = [];
            while (index < lines.length && /^\s*\|.+\|\s*$/.test(lines[index])) {
                rows.push(lines[index]);
                index += 1;
            }
            if (rows.length >= 2) html.push(renderTable(rows));
            else paragraph.push(rows[0]);
            continue;
        }
        paragraph.push(line);
        index += 1;
    }
    flushParagraph();
    return html.join("");
}

function money(value) {
    if (value == null) return "Rent not listed";
    return `$${Number(value).toLocaleString("en-US")}`;
}

function countLabel(value, singular, plural) {
    if (value == null) return null;
    const number = Number(value);
    const shown = Number.isInteger(number) ? String(number) : String(number);
    return `${shown} ${number === 1 ? singular : plural}`;
}

function unitKey(value) {
    return String(value).toLowerCase().replace(/[^a-z0-9]/g, "").replace(/0+(?=\d)/g, "");
}

function unitLine(listing) {
    if (!listing.unit) return "";
    const name = unitKey(listing.building || "");
    const unit = unitKey(listing.unit);
    if (unit && name.includes(unit)) return "";
    return `Unit ${listing.unit}`;
}

function metaLine(listing) {
    const parts = [];
    if (listing.neighborhood) parts.push(listing.neighborhood);
    const unit = unitLine(listing);
    if (unit) parts.push(unit);
    if (listing.beds === 0) parts.push("Studio");
    else {
        const beds = countLabel(listing.beds, "bed", "beds");
        if (beds) parts.push(beds);
    }
    const baths = countLabel(listing.baths, "bath", "baths");
    if (baths) parts.push(baths);
    if (listing.sqft) parts.push(`${Number(listing.sqft).toLocaleString("en-US")} sqft`);
    if (listing.deal_score != null) parts.push(`Deal ${Number(listing.deal_score).toFixed(1)}`);
    return parts.join(" · ");
}

function savedIds() {
    return new Set(state.favorites.map((listing) => listing.listing_id));
}

function withSaved(listings) {
    const saved = savedIds();
    return (listings || []).map((listing) => ({
        ...listing,
        favorited: saved.has(listing.listing_id),
    }));
}

function photo(listing, link) {
    const frame = link && listing.url ? document.createElement("a") : el("div");
    frame.className = "photo";
    if (link && listing.url) {
        frame.href = listing.url;
        frame.target = "_blank";
        frame.rel = "noopener noreferrer";
    }
    if (listing.image) {
        const img = document.createElement("img");
        img.src = listing.image;
        img.alt = "";
        img.decoding = "async";
        img.loading = "lazy";
        frame.append(img);
    }
    return frame;
}

function viewLink(listing) {
    if (!listing.url) return null;
    const link = document.createElement("a");
    link.className = "btn btn-primary";
    link.href = listing.url;
    link.target = "_blank";
    link.rel = "noopener noreferrer";
    link.textContent = "View listing";
    return link;
}

function heartButton(listing) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = listing.favorited ? "heart on" : "heart";
    button.dataset.listing = listing.listing_id;
    button.setAttribute("aria-pressed", listing.favorited ? "true" : "false");
    button.setAttribute("aria-label", listing.favorited ? "Remove saved listing" : "Save listing");
    button.innerHTML = HEART;
    button.addEventListener("click", (event) => {
        event.preventDefault();
        event.stopPropagation();
        const saved = button.getAttribute("aria-pressed") === "true";
        toggleFavorite(listing.listing_id, !saved);
    });
    return button;
}

function openLink(listing, className) {
    const node = listing.url ? document.createElement("a") : el("div");
    node.className = className;
    if (listing.url) {
        node.href = listing.url;
        node.target = "_blank";
        node.rel = "noopener noreferrer";
    }
    return node;
}

function gridCard(listing) {
    const card = el("article", "card");
    card.dataset.listing = listing.listing_id;
    const open = openLink(listing, "card-open");
    open.append(photo(listing));
    const body = el("div", "card-body");
    body.append(el("p", "rent", money(listing.rent)));
    body.append(el("p", "building", listing.building || "Listing"));
    body.append(el("p", "meta", metaLine(listing)));
    open.append(body);
    const actions = el("div", "card-actions");
    const view = viewLink(listing);
    if (view) actions.append(view);
    actions.append(heartButton(listing));
    card.append(open, actions);
    return card;
}

function featureCard(listing) {
    const card = el("article", "feature");
    card.dataset.listing = listing.listing_id;
    card.append(photo(listing, true));
    const body = el("div", "feature-body");
    body.append(el("p", "rent", money(listing.rent)));
    body.append(el("p", "building", listing.building || "Listing"));
    body.append(el("p", "meta", metaLine(listing)));
    if (listing.description) body.append(el("p", "desc", listing.description));
    const actions = el("div", "feature-actions");
    const view = viewLink(listing);
    if (view) actions.append(view);
    actions.append(heartButton(listing));
    body.append(actions);
    card.append(body);
    return card;
}

function savedRow(listing) {
    const row = el("article", "saved");
    row.dataset.listing = listing.listing_id;
    const open = openLink(listing, "saved-open");
    open.append(photo(listing));
    const text = el("div", "saved-text");
    text.append(el("p", "rent", money(listing.rent)));
    text.append(el("p", "building", listing.building || "Listing"));
    const bits = [listing.neighborhood];
    if (listing.beds === 0) bits.push("Studio");
    else {
        const beds = countLabel(listing.beds, "bed", "beds");
        if (beds) bits.push(beds);
    }
    text.append(el("p", "meta", bits.filter(Boolean).join(" · ")));
    open.append(text);
    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "btn btn-quiet";
    remove.textContent = "Remove";
    remove.addEventListener("click", () => toggleFavorite(listing.listing_id, false));
    row.append(open, remove);
    return row;
}

function argLine(args) {
    if (!args || typeof args !== "object") return "";
    return Object.entries(args)
        .filter(([, value]) => value != null && value !== "")
        .map(([key, value]) => `${key}: ${Array.isArray(value) ? value.join(", ") : value}`)
        .join(" · ");
}

function resultSummary(call) {
    let data = null;
    if (call.result) {
        try {
            data = JSON.parse(call.result);
        } catch {
            data = null;
        }
    }
    if (data && data.error) return "Error";
    if (data && Array.isArray(data.listings)) {
        const count = data.count != null ? data.count : data.listings.length;
        return `${count} listing${count === 1 ? "" : "s"}`;
    }
    if (data && Array.isArray(data.rows)) {
        const count = data.rows.length;
        return `${count} row${count === 1 ? "" : "s"}${data.truncated ? ", truncated" : ""}`;
    }
    if (data && Array.isArray(data.amenities)) return `${data.amenities.length} amenities`;
    if (data && Array.isArray(data.neighborhoods)) return `${data.neighborhoods.length} neighborhoods`;
    if (Array.isArray(data)) return `${data.length} fields`;
    return "Done";
}

function pretty(result) {
    if (!result) return "";
    try {
        return JSON.stringify(JSON.parse(result), null, 2);
    } catch {
        return String(result);
    }
}

const STEP_LABELS = {
    search_listings: "Searching listings",
    cheapest_in_neighborhood: "Finding the cheapest listings",
    get_deals: "Ranking deals",
    best_deals_in_neighborhood: "Ranking the best deals",
    get_building_amenities: "Reading building amenities",
    compare_neighborhoods: "Comparing neighborhoods",
    summarize_neighborhood: "Summarizing the neighborhood",
    read_mappings: "Reading the table columns",
    get_enum_values: "Checking allowed values",
    run_sql: "Running a query",
};

function stepLabel(name) {
    return STEP_LABELS[name] || name || "Working";
}

function stepsSummary(calls) {
    if (!calls || calls.length === 0) return "Steps";
    if (calls.length === 1) return stepLabel(calls[0].name);
    return `${calls.length} steps`;
}

function toolBlock(call) {
    const block = el("section", "tool");
    if (call.running) block.classList.add("running");
    const title = el("p", "tool-name");
    title.append(stepLabel(call.name));
    const key = el("span", "tool-key", call.name || "");
    title.append(key);
    block.append(title);
    if (call.name === "run_sql" && call.args && call.args.query) {
        block.append(el("pre", "sql", call.args.query));
    } else {
        const line = argLine(call.args);
        if (line) block.append(el("p", "tool-args", line));
    }
    if (call.running) {
        block.append(el("p", "tool-status", "Running"));
        return block;
    }
    const details = document.createElement("details");
    const summary = document.createElement("summary");
    summary.textContent = resultSummary(call);
    details.append(summary);
    // Tool payloads can be large. Format them only if the step is opened.
    details.addEventListener("toggle", () => {
        if (!details.open || details.querySelector(".tool-result")) return;
        details.append(el("pre", "tool-result", pretty(call.result)));
    });
    block.append(details);
    return block;
}

function toolsFold(calls) {
    const details = document.createElement("details");
    details.className = "tools-fold";
    const summary = document.createElement("summary");
    summary.textContent = stepsSummary(calls);
    const tools = el("div", "tools");
    calls.forEach((call) => tools.append(toolBlock(call)));
    details.append(summary, tools);
    return details;
}

function listingNames(listing) {
    return [listing.building, listing.unit ? `${listing.building || ""} #${listing.unit}` : ""]
        .map((name) => String(name || "").trim().toLowerCase())
        .filter((name) => name.length >= 6);
}

function mentionsListing(listing, text) {
    const hay = text.toLowerCase();
    return listingNames(listing).some((name) => hay.includes(name));
}

function blockListing(block, listings, used) {
    const text = block.textContent || "";
    const hits = listings.filter((listing) => !used.has(listing.listing_id) && mentionsListing(listing, text));
    if (hits.length !== 1) return null;
    const listing = hits[0];
    const claimedBelow = [...block.querySelectorAll("li, p")].some((child) => {
        const nested = listings.filter((item) => !used.has(item.listing_id) && mentionsListing(item, child.textContent || ""));
        return nested.length === 1 && nested[0].listing_id === listing.listing_id;
    });
    return claimedBelow ? null : listing;
}

function mountInline(block, listing) {
    if (block.tagName === "P") {
        const host = el("div", block.className);
        while (block.firstChild) host.append(block.firstChild);
        block.replaceWith(host);
        block = host;
    }
    const body = el("div", "unit-body");
    const copy = el("div", "unit-copy");
    const head = el("div", "unit-head");
    const title = el("div", "unit-title");
    while (block.firstChild) copy.append(block.firstChild);
    while (copy.firstChild && copy.firstChild.nodeName !== "UL" && copy.firstChild.nodeName !== "OL") {
        title.append(copy.firstChild);
    }
    head.append(title);
    const view = viewLink(listing);
    if (view) {
        view.classList.add("btn-inline");
        head.append(view);
    }
    head.append(heartButton(listing));
    copy.prepend(head);
    if (listing.image) {
        const shot = photo(listing, true);
        shot.classList.add("inline-photo");
        shot.setAttribute("aria-label", listing.building ? `Photo of ${listing.building}` : "Listing photo");
        body.append(shot);
    }
    body.append(copy);
    block.classList.add("unit-line");
    block.append(body);
}

function attachInlineListings(answer, listings) {
    const used = new Set();
    answer.querySelectorAll("li, p").forEach((block) => {
        if (block.closest(".unit-line")) return;
        const listing = blockListing(block, listings, used);
        if (!listing) return;
        used.add(listing.listing_id);
        mountInline(block, listing);
    });
    return used;
}

function renderTurn(turn) {
    const section = el("article", "turn");
    if (turn.user) section.append(el("p", "user", turn.user));
    if (turn.pending) {
        const live = el("div", "tools-live");
        const label = el("p", "tools-label", "Thinking");
        const list = el("div", "tools");
        live.append(label, list);
        section.append(live);
        section._liveLabel = label;
        section._liveList = list;
        return section;
    }
    if (turn.tool_calls && turn.tool_calls.length) section.append(toolsFold(turn.tool_calls));
    const answer = turn.response ? el("div", "answer") : null;
    if (answer) answer.innerHTML = renderMarkdown(turn.response);
    const listings = withSaved(turn.listings);
    const inline = listings.length > 1 && answer ? attachInlineListings(answer, listings) : new Set();
    if (inline.size) {
        section.append(answer);
        const rest = listings.filter((listing) => !inline.has(listing.listing_id));
        if (rest.length > 1) {
            const row = el("div", "cards cards-inline");
            rest.forEach((listing) => row.append(gridCard(listing)));
            section.append(row);
        } else if (rest.length === 1) {
            section.append(featureCard(rest[0]));
        }
        return section;
    }
    if ((turn.display === "grid" || turn.display === "paired") && listings.length) {
        const split = el("div", "result-split");
        if (answer) split.append(answer);
        const grid = el("div", "cards");
        listings.forEach((listing) => grid.append(gridCard(listing)));
        split.append(grid);
        section.append(split);
    } else {
        if (answer) section.append(answer);
        if (turn.display === "single" && listings[0]) section.append(featureCard(listings[0]));
    }
    return section;
}

function readEvents(buffer) {
    const chunks = buffer.split("\n\n");
    const rest = chunks.pop();
    const events = [];
    chunks.forEach((chunk) => {
        const data = chunk
            .split("\n")
            .filter((line) => line.startsWith("data:"))
            .map((line) => line.slice(5).trim())
            .join("\n");
        if (data) events.push(JSON.parse(data));
    });
    return { events, rest };
}

function applyStreamEvent(pending, event) {
    if (event.type === "status") {
        pending._liveLabel.textContent = event.text || "Thinking";
        return;
    }
    if (event.type === "tool_start") {
        pending._liveLabel.textContent = stepLabel(event.name);
        const row = toolBlock({ name: event.name, args: event.args, running: true });
        row.dataset.index = String(event.index);
        pending._liveList.append(row);
        return;
    }
    if (event.type === "tool_end") {
        const row = pending._liveList.querySelector(`[data-index="${event.index}"]`);
        const done = toolBlock({ name: event.name, args: event.args, result: event.result });
        done.dataset.index = String(event.index);
        if (row) row.replaceWith(done);
        else pending._liveList.append(done);
    }
}

function renderEmpty() {
    log.replaceChildren();
    const empty = el("div", "empty");
    empty.append(el("p", "empty-lead", "Ask about a listing, a rent, or a neighborhood."));
    const list = el("div", "prompts");
    PROMPTS.forEach((prompt) => {
        const button = document.createElement("button");
        button.type = "button";
        button.className = "btn btn-secondary prompt";
        button.textContent = prompt;
        button.addEventListener("click", () => send(prompt));
        list.append(button);
    });
    empty.append(list);
    log.append(empty);
}

function showFromTop(node) {
    const box = logScroll.getBoundingClientRect();
    const rect = node.getBoundingClientRect();
    logScroll.scrollTop += rect.top - box.top - 8;
}

function renderStatus(text) {
    log.replaceChildren(el("p", "pending", text));
}

function renderThread(turns) {
    log.replaceChildren();
    if (!turns.length) {
        renderEmpty();
        return;
    }
    const fragment = document.createDocumentFragment();
    turns.forEach((turn) => fragment.append(renderTurn(turn)));
    log.append(fragment);
    const last = log.querySelector(".turn:last-child");
    if (last) showFromTop(last);
}

function renderTop() {
    const current = state.threads.find((thread) => thread.session_id === state.sessionId);
    topTitle.textContent = current ? current.title : "New chat";
}

function renderRail() {
    const scroll = railBody.scrollTop;
    railBody.replaceChildren();
    const head = el("div", "rail-head");
    if (state.tab === "chats") {
        const button = document.createElement("button");
        button.type = "button";
        button.className = "btn btn-secondary";
        button.textContent = "New chat";
        button.addEventListener("click", newChat);
        head.append(button);
        railBody.append(head);
        if (!state.threads.length) {
            railBody.append(el("p", "rail-note", "No chats yet."));
            railBody.scrollTop = scroll;
            return;
        }
        const list = el("div", "thread-list");
        state.threads.forEach((thread) => {
            const row = el("div", "thread-row");
            const open = document.createElement("button");
            open.type = "button";
            open.className = "thread-open";
            open.dataset.session = thread.session_id;
            open.textContent = thread.title;
            if (thread.session_id === state.sessionId) open.setAttribute("aria-current", "true");
            open.addEventListener("click", () => openThread(thread.session_id));
            const remove = document.createElement("button");
            remove.type = "button";
            remove.className = "btn btn-quiet";
            remove.textContent = "Delete";
            remove.addEventListener("click", () => deleteThread(thread.session_id));
            row.append(open, remove);
            list.append(row);
        });
        railBody.append(list);
        railBody.scrollTop = scroll;
        return;
    }
    const select = document.createElement("select");
    select.className = "sort";
    select.setAttribute("aria-label", "Sort saved listings");
    SORTS.forEach(([value, label]) => {
        const option = document.createElement("option");
        option.value = value;
        option.textContent = label;
        if (value === state.sort) option.selected = true;
        select.append(option);
    });
    select.addEventListener("change", async () => {
        state.sort = select.value;
        await loadFavorites();
        renderRail();
        syncHearts();
    });
    head.append(select);
    railBody.append(head);
    if (!state.favorites.length) {
        railBody.append(el("p", "rail-note", "No saved listings yet. Save a unit from a reply to keep it here."));
        railBody.scrollTop = scroll;
        return;
    }
    const list = el("div", "saved-list");
    state.favorites.forEach((listing) => list.append(savedRow(listing)));
    railBody.append(list);
    railBody.scrollTop = scroll;
}

function syncHearts() {
    const saved = savedIds();
    document.querySelectorAll(".heart").forEach((button) => {
        const on = saved.has(button.dataset.listing);
        button.classList.toggle("on", on);
        button.setAttribute("aria-pressed", on ? "true" : "false");
        button.setAttribute("aria-label", on ? "Remove saved listing" : "Save listing");
    });
}

function setTab(tab) {
    state.tab = tab;
    tabChats.setAttribute("aria-selected", tab === "chats" ? "true" : "false");
    tabSaved.setAttribute("aria-selected", tab === "saved" ? "true" : "false");
    renderRail();
}

function showChats() {
    const changed = state.tab !== "chats";
    state.tab = "chats";
    tabChats.setAttribute("aria-selected", "true");
    tabSaved.setAttribute("aria-selected", "false");
    if (changed) renderRail();
}

function closeRail() {
    rail.classList.remove("open");
    scrim.hidden = true;
}

function setBusy(busy) {
    state.sending = busy;
    sendBtn.disabled = busy;
    sendBtn.textContent = busy ? "Sending" : "Send";
    document.querySelectorAll(".prompt").forEach((button) => {
        button.disabled = busy;
    });
}

function markCurrentThread() {
    railBody.querySelectorAll(".thread-open").forEach((button) => {
        if (button.dataset.session === state.sessionId) button.setAttribute("aria-current", "true");
        else button.removeAttribute("aria-current");
    });
}

function refreshThreads() {
    loadThreads()
        .then(() => {
            renderTop();
            if (state.tab === "chats") renderRail();
        })
        .catch(() => {});
}

async function loadThreads() {
    const response = await fetch("/threads");
    const data = await response.json();
    state.threads = data.threads || [];
}

async function loadFavorites() {
    const response = await fetch(`/favorites?sort=${encodeURIComponent(state.sort)}`);
    const data = await response.json();
    state.sort = data.sort || state.sort;
    state.favorites = data.listings || [];
}

function newChat() {
    showChats();
    state.sessionId = null;
    state.title = "New chat";
    sessionStorage.removeItem(STORE);
    renderEmpty();
    renderTop();
    renderRail();
    closeRail();
    input.focus();
}

function applyLoaded(data) {
    state.sessionId = data.session_id;
    state.title = data.title || "Chat";
    sessionStorage.setItem(STORE, state.sessionId);
    renderThread(data.turns || []);
    renderTop();
    markCurrentThread();
}

async function openThread(sessionId) {
    if (sessionId === state.sessionId && log.querySelector(".turn")) {
        closeRail();
        return;
    }
    state.sessionId = sessionId;
    sessionStorage.setItem(STORE, sessionId);
    markCurrentThread();
    renderTop();
    renderStatus("Opening chat");
    closeRail();
    const response = await fetch(`/threads/${encodeURIComponent(sessionId)}`);
    if (state.sessionId !== sessionId) return;
    if (!response.ok) {
        sessionStorage.removeItem(STORE);
        newChat();
        return;
    }
    const data = await response.json();
    if (state.sessionId !== sessionId) return;
    applyLoaded(data);
}

async function deleteThread(sessionId) {
    const snapshot = state.threads.slice();
    state.threads = state.threads.filter((thread) => thread.session_id !== sessionId);
    if (state.sessionId === sessionId) {
        state.sessionId = null;
        sessionStorage.removeItem(STORE);
        renderEmpty();
        renderTop();
    }
    renderRail();
    const response = await fetch(`/clear?session_id=${encodeURIComponent(sessionId)}`, { method: "POST" });
    if (!response.ok) {
        state.threads = snapshot;
        renderRail();
    }
}

async function toggleFavorite(listingId, shouldSave) {
    const snapshot = state.favorites.map((listing) => ({ ...listing }));
    if (shouldSave) {
        if (!state.favorites.some((listing) => listing.listing_id === listingId)) {
            state.favorites = [{ listing_id: listingId }, ...state.favorites];
        }
    } else {
        state.favorites = state.favorites.filter((listing) => listing.listing_id !== listingId);
    }
    syncHearts();
    if (state.tab === "saved") renderRail();
    const response = shouldSave
        ? await fetch("/favorites", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ listing_id: listingId }),
        })
        : await fetch(`/favorites/${encodeURIComponent(listingId)}`, { method: "DELETE" });
    if (!response.ok) {
        state.favorites = snapshot;
        syncHearts();
        if (state.tab === "saved") renderRail();
        return;
    }
    if (shouldSave) {
        const card = await response.json().catch(() => null);
        if (card && card.listing_id) {
            state.favorites = [
                card,
                ...state.favorites.filter((listing) => listing.listing_id !== listingId),
            ];
        }
    }
    syncHearts();
    if (shouldSave && state.tab === "saved") renderRail();
    if (shouldSave && state.sort !== "recent") {
        loadFavorites()
            .then(() => {
                syncHearts();
                if (state.tab === "saved") renderRail();
            })
            .catch(() => {});
    }
}

async function send(text) {
    const message = text.trim();
    if (!message || state.sending) return;
    showChats();
    setBusy(true);
    input.value = "";
    input.style.height = "auto";
    if (!log.querySelector(".turn")) log.replaceChildren();
    const pending = renderTurn({ user: message, pending: true });
    log.append(pending);
    showFromTop(pending);
    let keepFocus = true;
    try {
        const response = await fetch("/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ message, session_id: state.sessionId }),
        });
        if (!response.ok || !response.body) {
            const data = await response.json().catch(() => ({}));
            throw new Error(data.detail || response.statusText);
        }
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";
        let finished = null;
        while (!finished) {
            const chunk = await reader.read();
            if (chunk.done) break;
            buffer += decoder.decode(chunk.value, { stream: true });
            const parsed = readEvents(buffer);
            buffer = parsed.rest;
            parsed.events.forEach((event) => {
                if (event.type === "done") finished = event;
                else applyStreamEvent(pending, event);
            });
        }
        if (!finished) throw new Error("The reply ended before an answer arrived.");
        if (!log.contains(pending)) {
            keepFocus = false;
            refreshThreads();
            return;
        }
        state.sessionId = finished.session_id;
        sessionStorage.setItem(STORE, state.sessionId);
        const done = renderTurn({
            user: message,
            response: finished.response || "",
            tool_calls: finished.tool_calls || [],
            display: finished.display || "none",
            listings: finished.listings || [],
        });
        pending.replaceWith(done);
        const answer = done.querySelector(".answer, .feature, .cards") || done;
        showFromTop(answer);
        refreshThreads();
    } catch (error) {
        if (!log.contains(pending)) keepFocus = false;
        else {
            pending.replaceWith(renderTurn({
                user: message,
                response: `Something went wrong: ${error.message}`,
                tool_calls: [],
                display: "none",
                listings: [],
            }));
        }
    } finally {
        setBusy(false);
        if (keepFocus) input.focus();
    }
}

tabChats.addEventListener("click", () => setTab("chats"));
tabSaved.addEventListener("click", () => setTab("saved"));
document.getElementById("menu").addEventListener("click", () => {
    rail.classList.add("open");
    scrim.hidden = false;
});
scrim.addEventListener("click", closeRail);
document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") closeRail();
});

form.addEventListener("submit", (event) => {
    event.preventDefault();
    send(input.value);
});

input.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        form.requestSubmit();
    }
});

input.addEventListener("input", () => {
    input.style.height = "auto";
    input.style.height = `${Math.min(input.scrollHeight, 160)}px`;
});

async function init() {
    const stored = sessionStorage.getItem(STORE);
    const pendingThread = stored
        ? fetch(`/threads/${encodeURIComponent(stored)}`).then((response) => (
            response.ok ? response.json() : null
        )).catch(() => null)
        : null;
    renderStatus("Loading");
    try {
        await Promise.all([loadThreads(), loadFavorites()]);
    } catch {
        state.threads = [];
        state.favorites = [];
    }
    renderRail();
    if (pendingThread && state.threads.some((thread) => thread.session_id === stored)) {
        state.sessionId = stored;
        markCurrentThread();
        renderTop();
        const data = await pendingThread;
        if (state.sessionId !== stored) return;
        if (data && data.session_id === stored) {
            applyLoaded(data);
            return;
        }
    }
    newChat();
}

init();
