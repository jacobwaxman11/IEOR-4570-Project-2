const PROMPTS = [
    "Find 1-bedrooms in Chelsea under $6,500 with a doorman, an elevator, and a dishwasher, at least 600 sq ft.",
    "What's the best 1-bedroom deal on the Upper West Side at or under $5,000? Include the address, listed rent, predicted rent, and deal score.",
    "Compare Chelsea and the West Village for 1-bedrooms under $6,500: median rent, typical square feet, and the most common amenities.",
    "In Hell's Kitchen, chart average rent for studios, 1-bedrooms, and 2-bedrooms, and show how many listings there are of each.",
    "Show a map of Manhattan neighborhoods colored by average rent, and a bar chart of those averages from lowest to highest.",
    "What amenities does 95 Wall Street have, and what's its cheapest 1-bedroom? Include the unit, square feet, and monthly rent.",
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
    account: null,
    mapsKey: "",
    clientId: "",
    mapsReady: null,
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
    if (listing.street) parts.push(listing.street);
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
        toggleFavorite(listing.listing_id, !saved, listing);
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
    if (data && Array.isArray(data.columns) && !Array.isArray(data.rows)) {
        return `${data.columns.length} fields`;
    }
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
    search_places: "Searching places",
    get_directions: "Checking the route",
    commute_from_listing: "Checking the commute",
    neighborhood_activities: "Looking up things to do",
    show_listing_map: "Locating the unit",
    average_rent_by_neighborhood: "Averaging rent by neighborhood",
    average_rent_by_unit_type: "Averaging rent by apartment type",
    listing_mix: "Counting the listing mix",
    neighborhood_price_map: "Mapping neighborhood rents",
    save_profile: "Saving the renter profile",
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

function moneyTick(value) {
    const number = Number(value);
    if (!Number.isFinite(number)) return "";
    return `$${Math.round(number).toLocaleString("en-US")}`;
}

function chartLabel(point, unit) {
    if (unit === "usd") return moneyTick(point.value);
    return Number(point.value).toLocaleString("en-US");
}

function barChart(spec) {
    const points = (spec.points || []).filter((point) => point && point.value != null);
    if (!points.length) return null;
    const width = 420;
    const row = 28;
    const labelW = 148;
    const height = points.length * row + 8;
    const max = Math.max(...points.map((point) => Number(point.value) || 0), 1);
    const bars = points.map((point, index) => {
        const y = 8 + index * row;
        const length = Math.max(2, ((Number(point.value) || 0) / max) * (width - labelW - 72));
        return `<text x="0" y="${y + 16}" font-size="12">${escapeHtml(point.label)}</text>
            <rect x="${labelW}" y="${y + 4}" width="${length}" height="14" fill="#234737"></rect>
            <text x="${labelW + length + 6}" y="${y + 16}" font-size="12">${escapeHtml(chartLabel(point, spec.unit))}</text>`;
    }).join("");
    const wrap = el("figure", "chart");
    wrap.append(el("figcaption", "chart-title", spec.title || "Chart"));
    wrap.insertAdjacentHTML("beforeend", `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="${escapeHtml(spec.title || "Bar chart")}">${bars}</svg>`);
    return wrap;
}

function pieChart(spec) {
    const points = (spec.points || []).filter((point) => Number(point.value) > 0);
    if (!points.length) return null;
    const total = points.reduce((sum, point) => sum + Number(point.value), 0);
    const colors = ["#234737", "#6d8f78", "#b08968", "#8c4a3a", "#3d4c5c", "#c4a574", "#5c6b4a", "#7a6a58"];
    let angle = -Math.PI / 2;
    const slices = points.map((point, index) => {
        const share = Number(point.value) / total;
        const next = angle + share * Math.PI * 2;
        const large = share > 0.5 ? 1 : 0;
        const x1 = 80 + Math.cos(angle) * 62;
        const y1 = 80 + Math.sin(angle) * 62;
        const x2 = 80 + Math.cos(next) * 62;
        const y2 = 80 + Math.sin(next) * 62;
        const path = share >= 0.999
            ? `<circle cx="80" cy="80" r="62" fill="${colors[index % colors.length]}"></circle>`
            : `<path d="M80 80 L${x1} ${y1} A62 62 0 ${large} 1 ${x2} ${y2} Z" fill="${colors[index % colors.length]}"></path>`;
        angle = next;
        return path;
    }).join("");
    const legend = points.map((point, index) => {
        const pct = Math.round((Number(point.value) / total) * 100);
        return `<div><span style="color:${colors[index % colors.length]}">●</span> ${escapeHtml(point.label)} ${pct}%</div>`;
    }).join("");
    const wrap = el("figure", "chart");
    wrap.append(el("figcaption", "chart-title", spec.title || "Listing mix"));
    wrap.insertAdjacentHTML("beforeend", `<svg class="pie" viewBox="0 0 160 160" role="img" aria-label="${escapeHtml(spec.title || "Pie chart")}">${slices}</svg><div class="meta">${legend}</div>`);
    return wrap;
}

function appendCharts(section, charts) {
    (charts || []).forEach((spec) => {
        const node = spec.type === "pie" ? pieChart(spec) : spec.type === "bar" ? barChart(spec) : null;
        if (node) section.append(node);
    });
}

function loadMaps() {
    if (!state.mapsKey) return Promise.resolve(false);
    if (window.google && window.google.maps) return Promise.resolve(true);
    if (state.mapsReady) return state.mapsReady;
    state.mapsReady = new Promise((resolve) => {
        const script = document.createElement("script");
        script.src = `https://maps.googleapis.com/maps/api/js?key=${encodeURIComponent(state.mapsKey)}`;
        script.async = true;
        script.onload = () => resolve(true);
        script.onerror = () => resolve(false);
        document.head.append(script);
    });
    return state.mapsReady;
}

function decodePolyline(encoded) {
    const points = [];
    let index = 0;
    let lat = 0;
    let lng = 0;
    while (index < encoded.length) {
        let result = 0;
        let shift = 0;
        let byte = 0;
        do {
            byte = encoded.charCodeAt(index++) - 63;
            result |= (byte & 0x1f) << shift;
            shift += 5;
        } while (byte >= 0x20);
        lat += (result & 1) ? ~(result >> 1) : result >> 1;
        result = 0;
        shift = 0;
        do {
            byte = encoded.charCodeAt(index++) - 63;
            result |= (byte & 0x1f) << shift;
            shift += 5;
        } while (byte >= 0x20);
        lng += (result & 1) ? ~(result >> 1) : result >> 1;
        points.push({ lat: lat / 1e5, lng: lng / 1e5 });
    }
    return points;
}

function rentColor(value, min, max) {
    const span = max - min || 1;
    const t = (Number(value) - min) / span;
    return `hsl(${Math.round(140 - t * 140)} 42% 38%)`;
}

function appendMap(section, map) {
    if (!map) return;
    const pins = map.pins || [];
    const circles = map.circles || [];
    const routes = map.routes || [];
    if (!pins.length && !circles.length && !routes.length) return;
    const block = el("div", "map-block");
    const title = circles.length ? "Average rent by neighborhood" : routes.length ? "Commute" : "Map";
    block.append(el("p", "map-title", title));
    const canvas = el("div", "map-canvas");
    block.append(canvas);
    section.append(block);
    loadMaps().then((ready) => {
        if (!ready || !window.google) {
            canvas.replaceWith(el("p", "meta", "Map unavailable. Set GOOGLE_MAPS_API_KEY and allow this page's origin."));
            return;
        }
        const first = pins[0] || circles[0] || { lat: 40.75, lng: -73.98 };
        const view = new google.maps.Map(canvas, {
            center: { lat: Number(first.lat), lng: Number(first.lng) },
            zoom: circles.length ? 12 : 13,
            mapTypeControl: false,
            streetViewControl: false,
        });
        const bounds = new google.maps.LatLngBounds();
        pins.forEach((pin) => {
            const position = { lat: Number(pin.lat), lng: Number(pin.lng) };
            bounds.extend(position);
            new google.maps.Marker({ map: view, position, title: pin.label || "" });
        });
        if (circles.length) {
            const values = circles.map((point) => Number(point.value) || 0);
            const min = Math.min(...values);
            const max = Math.max(...values);
            circles.forEach((point) => {
                const position = { lat: Number(point.lat), lng: Number(point.lng) };
                bounds.extend(position);
                new google.maps.Circle({
                    map: view,
                    center: position,
                    radius: 450,
                    strokeWeight: 0,
                    fillColor: rentColor(point.value, min, max),
                    fillOpacity: 0.75,
                });
                new google.maps.Marker({
                    map: view,
                    position,
                    label: { text: point.label || "", fontSize: "11px" },
                    icon: { path: google.maps.SymbolPath.CIRCLE, scale: 0 },
                });
            });
        }
        routes.forEach((route) => {
            if (!route.polyline) return;
            const path = decodePolyline(route.polyline);
            path.forEach((point) => bounds.extend(point));
            new google.maps.Polyline({ map: view, path, strokeColor: "#234737", strokeWeight: 4 });
        });
        if (!bounds.isEmpty()) view.fitBounds(bounds, 32);
    });
}

function appendFollowups(section, prompts) {
    if (!prompts || !prompts.length) return;
    const row = el("div", "followups");
    prompts.forEach((prompt) => {
        const button = document.createElement("button");
        button.type = "button";
        button.className = "btn btn-secondary";
        button.textContent = prompt;
        button.addEventListener("click", () => send(prompt));
        row.append(button);
    });
    section.append(row);
}

function finishTurn(section, turn) {
    appendCharts(section, turn.charts);
    appendMap(section, turn.map);
    appendFollowups(section, turn.follow_ups);
    return section;
}

function renderApproval(section, event) {
    const box = el("div", "approval");
    const fields = event.arguments || {};
    const lines = Object.entries(fields)
        .filter(([, value]) => value != null && value !== "")
        .map(([key, value]) => `${key.replaceAll("_", " ")}: ${value}`);
    box.append(el("p", null, "Save this on your account?"));
    if (lines.length) box.append(el("p", "meta", lines.join(" · ")));
    const actions = el("div", "approval-actions");
    const approve = document.createElement("button");
    approve.type = "button";
    approve.className = "btn btn-primary";
    approve.textContent = "Approve";
    approve.addEventListener("click", () => resume(true, section));
    const reject = document.createElement("button");
    reject.type = "button";
    reject.className = "btn btn-secondary";
    reject.textContent = "Reject";
    reject.addEventListener("click", () => resume(false, section));
    actions.append(approve, reject);
    box.append(actions);
    section.append(box);
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
        return finishTurn(section, turn);
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
    return finishTurn(section, turn);
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
    if (tab !== "saved") return;
    loadFavorites()
        .then(() => {
            syncHearts();
            if (state.tab === "saved") renderRail();
        })
        .catch(() => {});
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

const favoriteWrites = new Map();
let favoriteEpoch = 0;
let favoriteLoad = 0;

function mergeListing(local, remote) {
    const merged = { ...(local || {}), ...(remote || {}), favorited: true };
    if (!local) return merged;
    for (const key of ["building", "unit", "neighborhood", "beds", "baths", "sqft", "rent", "url", "deal_score", "image", "description"]) {
        if ((merged[key] == null || merged[key] === "") && local[key] != null && local[key] !== "") {
            merged[key] = local[key];
        }
    }
    return merged;
}

function putFavorite(card) {
    const rest = state.favorites.filter((listing) => listing.listing_id !== card.listing_id);
    state.favorites = [card, ...rest];
}

function dropFavorite(listingId) {
    state.favorites = state.favorites.filter((listing) => listing.listing_id !== listingId);
}

async function loadFavorites() {
    const request = ++favoriteLoad;
    const epoch = favoriteEpoch;
    const response = await fetch(`/favorites?sort=${encodeURIComponent(state.sort)}`, { cache: "no-store" });
    if (!response.ok) throw new Error("Couldn't load saved listings");
    const data = await response.json();
    if (request !== favoriteLoad || epoch !== favoriteEpoch) return;
    state.sort = data.sort || state.sort;
    const incoming = data.listings || [];
    const incomingIds = new Set(incoming.map((listing) => listing.listing_id));
    const pending = [];
    const hidden = new Set();
    for (const [listingId, write] of favoriteWrites) {
        if (write.intent === "remove") {
            hidden.add(listingId);
            continue;
        }
        if (!incomingIds.has(listingId)) {
            const local = state.favorites.find((listing) => listing.listing_id === listingId);
            if (local) pending.push(local);
        }
    }
    state.favorites = [
        ...pending,
        ...incoming
            .filter((listing) => !hidden.has(listing.listing_id))
            .map((listing) => mergeListing(
                state.favorites.find((item) => item.listing_id === listing.listing_id),
                listing,
            )),
    ];
}

function warmMappings() {
    // The column read is slow on a new thread. Start it while the chat is still empty.
    // The running row is drawn later, with the other tool steps, after Send.
    fetch("/mappings/warm", { method: "POST" }).catch(() => {});
}

function newChat() {
    warmMappings();
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

async function toggleFavorite(listingId, shouldSave, listing) {
    const previous = state.favorites.find((item) => item.listing_id === listingId) || null;
    const generation = (favoriteWrites.get(listingId)?.generation || 0) + 1;
    const mine = () => favoriteWrites.get(listingId)?.generation === generation;
    favoriteEpoch += 1;
    favoriteWrites.set(listingId, { generation, intent: shouldSave ? "save" : "remove" });
    // Prod confirms the save through BigQuery. The clicked card already has
    // rent and a photo, so Saved shows that instead of an empty placeholder.
    if (shouldSave) {
        const seed = mergeListing(previous, listing || { listing_id: listingId });
        seed.listing_id = listingId;
        putFavorite(seed);
    } else {
        dropFavorite(listingId);
    }
    syncHearts();
    if (state.tab === "saved") renderRail();

    let failed = false;
    try {
        const response = shouldSave
            ? await fetch("/favorites", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ listing_id: listingId }),
            })
            : await fetch(`/favorites/${encodeURIComponent(listingId)}`, { method: "DELETE" });
        if (!mine()) return;
        if (!response.ok) {
            failed = true;
        } else if (shouldSave) {
            const card = await response.json().catch(() => null);
            if (!mine()) return;
            if (card && card.listing_id) {
                const merged = mergeListing(listing || previous, card);
                merged.listing_id = card.listing_id;
                putFavorite(merged);
            }
        }
    } catch {
        if (!mine()) return;
        failed = true;
    }
    if (!mine()) return;
    favoriteWrites.delete(listingId);
    favoriteEpoch += 1;
    try {
        await loadFavorites();
    } catch {
        if (failed) {
            if (previous) putFavorite(previous);
            else dropFavorite(listingId);
        }
    }
    syncHearts();
    if (state.tab === "saved") renderRail();
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
        if (response.status === 401) {
            showGate();
            throw new Error("Sign in required");
        }
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
                else if (event.type === "approval") {
                    pending._approval = event;
                    renderApproval(pending, event);
                } else applyStreamEvent(pending, event);
            });
        }
        if (!finished) {
            if (pending._approval) {
                state.sessionId = pending._approval.session_id || state.sessionId;
                return;
            }
            throw new Error("The reply ended before an answer arrived.");
        }
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
            charts: finished.charts || [],
            map: finished.map || null,
            follow_ups: finished.follow_ups || [],
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

async function resume(approved, section) {
    if (state.sending) return;
    setBusy(true);
    const box = section.querySelector(".approval");
    if (box) box.remove();
    let keepFocus = true;
    try {
        const response = await fetch("/chat/resume", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ session_id: state.sessionId, approved }),
        });
        if (response.status === 401) {
            showGate();
            throw new Error("Sign in required");
        }
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
                else if (event.type === "approval") {
                    section._approval = event;
                    renderApproval(section, event);
                } else if (section._liveList) applyStreamEvent(section, event);
            });
        }
        if (!finished) {
            if (section._approval) return;
            throw new Error("The reply ended before an answer arrived.");
        }
        state.sessionId = finished.session_id || state.sessionId;
        const user = section.querySelector(".user");
        const done = renderTurn({
            user: user ? user.textContent : "",
            response: finished.response || "",
            tool_calls: finished.tool_calls || [],
            display: finished.display || "none",
            listings: finished.listings || [],
            charts: finished.charts || [],
            map: finished.map || null,
            follow_ups: finished.follow_ups || [],
        });
        section.replaceWith(done);
        refreshThreads();
    } catch (error) {
        section.append(el("p", "meta", error.message));
    } finally {
        setBusy(false);
        if (keepFocus) input.focus();
    }
}

function renderAccount() {
    const node = document.getElementById("account");
    if (!node) return;
    node.replaceChildren();
    if (!state.account) return;
    node.append(document.createTextNode(state.account.email || state.account.name || "Signed in"));
    const out = document.createElement("button");
    out.type = "button";
    out.className = "btn btn-quiet";
    out.textContent = "Sign out";
    out.addEventListener("click", async () => {
        await fetch("/auth/logout", { method: "POST" });
        state.account = null;
        showGate();
    });
    node.append(document.createTextNode(" "));
    node.append(out);
}

function showGate() {
    const gate = document.getElementById("gate");
    const note = document.getElementById("gate-note");
    gate.hidden = false;
    if (!state.clientId) {
        note.textContent = "Set GOOGLE_CLIENT_ID in .env, then reload.";
        return;
    }
    note.textContent = "";
    const start = () => {
        google.accounts.id.initialize({
            client_id: state.clientId,
            callback: async (response) => {
                const saved = await fetch("/auth/google", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ credential: response.credential }),
                });
                if (!saved.ok) {
                    const data = await saved.json().catch(() => ({}));
                    note.textContent = data.detail || "Sign-in failed";
                    return;
                }
                state.account = await saved.json();
                gate.hidden = true;
                renderAccount();
                bootChat();
            },
        });
        const slot = document.getElementById("google-button");
        slot.replaceChildren();
        google.accounts.id.renderButton(slot, {
            theme: "outline",
            size: "large",
            width: 280,
        });
    };
    if (window.google && window.google.accounts) {
        start();
        return;
    }
    const script = document.createElement("script");
    script.src = "https://accounts.google.com/gsi/client";
    script.async = true;
    script.onload = start;
    script.onerror = () => {
        note.textContent = "Google sign-in failed to load.";
    };
    document.head.append(script);
}

async function bootChat() {
    warmMappings();
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
    renderAccount();
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

async function init() {
    const config = await fetch("/auth/config").then((response) => response.json()).catch(() => ({}));
    state.mapsKey = config.maps_api_key || "";
    state.clientId = config.client_id || "";
    const me = await fetch("/auth/me");
    if (!me.ok) {
        renderEmpty();
        showGate();
        return;
    }
    state.account = await me.json();
    renderAccount();
    bootChat();
}

init();
