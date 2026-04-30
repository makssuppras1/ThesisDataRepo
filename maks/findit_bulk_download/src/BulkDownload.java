import java.io.IOException;
import java.net.URI;
import java.net.URLDecoder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.nio.file.StandardOpenOption;
import java.time.Duration;
import java.time.Instant;
import java.time.ZonedDateTime;
import java.time.format.DateTimeFormatter;
import java.time.format.DateTimeParseException;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.OptionalLong;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import java.net.http.HttpClient;
import java.net.http.HttpHeaders;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;

/**
 * Bulk-download FindIt resolver URLs (one per line) with conservative throttling,
 * 429 exponential backoff (+ Retry-After), flat retries for network/5xx,
 * optional Cookie header file, atomic writes via .part, resume by final filename.
 *
 * <p>Authentication: export cookies after logging in to DTU FindIt (see cookies-howto.txt).
 */
public final class BulkDownload {

    private static final int MAX_429_ATTEMPTS = 10;
    private static final int MAX_NETWORK_ATTEMPTS = 10;
    private static final String UA = "FindItBulkDownload/1.0 (+https://github.com/)";

    public static void main(String[] args) throws Exception {
        Path input = null;
        Path outDir = Path.of("findit_downloads");
        Path cookieFile = null;
        long delayMs = 3_000;
        boolean force = false;
        int connectTimeoutSec = 30;
        int requestTimeoutMin = 15;

        for (int i = 0; i < args.length; i++) {
            String a = args[i];
            if ("--help".equals(a) || "-h".equals(a)) {
                usage();
                return;
            }
            if ("-o".equals(a) && i + 1 < args.length) {
                outDir = Path.of(args[++i]);
            } else if ("--delay-ms".equals(a) && i + 1 < args.length) {
                delayMs = Long.parseLong(args[++i]);
            } else if ("-c".equals(a) && i + 1 < args.length) {
                cookieFile = Path.of(args[++i]);
            } else if ("--force".equals(a)) {
                force = true;
            } else if ("--connect-timeout-sec".equals(a) && i + 1 < args.length) {
                connectTimeoutSec = Integer.parseInt(args[++i]);
            } else if ("--request-timeout-min".equals(a) && i + 1 < args.length) {
                requestTimeoutMin = Integer.parseInt(args[++i]);
            } else if (!a.startsWith("-")) {
                input = Path.of(a);
            } else {
                System.err.println("Unknown option: " + a);
                usage();
                System.exit(2);
            }
        }
        if (input == null) {
            System.err.println("Missing input file.");
            usage();
            System.exit(2);
        }

        String cookieHeader = cookieFile != null ? loadCookieHeader(cookieFile) : "";

        Files.createDirectories(outDir);

        HttpClient client = HttpClient.newBuilder()
                .followRedirects(HttpClient.Redirect.ALWAYS)
                .connectTimeout(Duration.ofSeconds(connectTimeoutSec))
                .build();

        List<String> lines = Files.readAllLines(input, StandardCharsets.UTF_8);
        Set<String> seenIds = new HashSet<>();
        int nDownloaded = 0;
        int nSkipDup = 0;
        int nSkipExists = 0;
        int nFailed = 0;

        for (String line : lines) {
            line = line.trim();
            if (line.isEmpty() || line.startsWith("#")) {
                continue;
            }
            String id = extractThesisId(line);
            if (id == null) {
                System.err.println("skip_unparseable: " + line.substring(0, Math.min(80, line.length())));
                continue;
            }
            if (!seenIds.add(id)) {
                System.out.println("skip_duplicate_id " + id);
                nSkipDup++;
                continue;
            }
            String ext = existingFinalExtension(outDir, id);
            if (ext != null && !force) {
                System.out.println("skip_exists " + id + "." + ext);
                nSkipExists++;
                continue;
            }
            if (force) {
                deletePartAndFinals(outDir, id);
            } else {
                deletePartOnly(outDir, id);
            }

            boolean ok = downloadOne(
                    client,
                    line,
                    cookieHeader,
                    outDir,
                    id,
                    Duration.ofMinutes(requestTimeoutMin),
                    force);
            if (ok) {
                nDownloaded++;
            } else {
                nFailed++;
            }
            Thread.sleep(delayMs + (long) (Math.random() * 1_000));
        }

        System.err.printf(
                Locale.ROOT,
                "done downloaded=%d skip_duplicate_id=%d skip_exists=%d failed=%s%n",
                nDownloaded,
                nSkipDup,
                nSkipExists,
                nFailed);
    }

    private static void usage() {
        System.err.println(
                "Usage (after: sh maks/findit_bulk_download/build.sh):\n"
                        + "  java -cp maks/findit_bulk_download/out BulkDownload PATH/TO/urls.txt\n"
                        + "\n"
                        + "Required: urls file — one resolver URL per line.\n"
                        + "Options:\n"
                        + "  -o DIR                 output directory (default: findit_downloads)\n"
                        + "  -c FILE                cookie line or Netscape cookie jar\n"
                        + "  --delay-ms N           ms pause after each item (default: 3000)\n"
                        + "  --force                overwrite existing files\n"
                        + "  --connect-timeout-sec N   (default: 30)\n"
                        + "  --request-timeout-min N (default: 15)\n"
                        + "\n"
                        + "Example (no angle brackets — use real paths):\n"
                        + "  java -cp maks/findit_bulk_download/out BulkDownload "
                        + "maks/findit_urls/dtu_theses_all_urls_deduped.txt -o ./downloads -c maks/cookies.txt\n");
    }

    /** Returns extension (pdf or bin) if a non-empty final file exists, else null. */
    private static String existingFinalExtension(Path outDir, String id) throws IOException {
        Path pdf = outDir.resolve(id + ".pdf");
        Path bin = outDir.resolve(id + ".bin");
        if (Files.isRegularFile(pdf) && Files.size(pdf) > 0) {
            return "pdf";
        }
        if (Files.isRegularFile(bin) && Files.size(bin) > 0) {
            return "bin";
        }
        return null;
    }

    private static void deletePartOnly(Path outDir, String id) throws IOException {
        Path part = outDir.resolve(id + ".part");
        Files.deleteIfExists(part);
    }

    private static void deletePartAndFinals(Path outDir, String id) throws IOException {
        deletePartOnly(outDir, id);
        Files.deleteIfExists(outDir.resolve(id + ".pdf"));
        Files.deleteIfExists(outDir.resolve(id + ".bin"));
    }

    private static final Pattern RFT_DAT_ID = Pattern.compile("\"id\"\\s*:\\s*\"([^\"]+)\"");

    static String extractThesisId(String url) {
        try {
            URI u = URI.create(url);
            String q = u.getRawQuery();
            if (q == null) {
                return null;
            }
            for (String pair : q.split("&")) {
                int eq = pair.indexOf('=');
                if (eq < 0) {
                    continue;
                }
                String key = URLDecoder.decode(pair.substring(0, eq), StandardCharsets.UTF_8);
                if (!"rft_dat".equals(key)) {
                    continue;
                }
                String enc = pair.substring(eq + 1);
                String json = URLDecoder.decode(enc, StandardCharsets.UTF_8);
                Matcher m = RFT_DAT_ID.matcher(json);
                if (m.find()) {
                    return m.group(1);
                }
            }
        } catch (Exception ignored) {
            return null;
        }
        return null;
    }

    static String loadCookieHeader(Path file) throws IOException {
        List<String> lines = Files.readAllLines(file, StandardCharsets.UTF_8);
        if (lines.isEmpty()) {
            return "";
        }
        boolean netscape = lines.stream().anyMatch(l -> l.startsWith("# Netscape HTTP Cookie File"));
        if (!netscape && lines.size() == 1 && !lines.get(0).contains("\t")) {
            return lines.get(0).trim();
        }
        Map<String, String> byName = new LinkedHashMap<>();
        for (String line : lines) {
            line = line.trim();
            if (line.isEmpty() || line.startsWith("#")) {
                continue;
            }
            String[] parts = line.split("\t", -1);
            if (parts.length < 7) {
                continue;
            }
            String domain = parts[0];
            if (!domain.contains("dtu.dk")) {
                continue;
            }
            String name = parts[parts.length - 2];
            String value = parts[parts.length - 1];
            byName.put(name, value);
        }
        StringBuilder sb = new StringBuilder();
        for (Map.Entry<String, String> e : byName.entrySet()) {
            if (sb.length() > 0) {
                sb.append("; ");
            }
            sb.append(e.getKey()).append('=').append(e.getValue());
        }
        return sb.toString();
    }

    static OptionalLong parseRetryAfterSeconds(HttpHeaders headers) {
        return headers.firstValue("Retry-After").map(BulkDownload::parseRetryAfterValue).orElseGet(OptionalLong::empty);
    }

    private static OptionalLong parseRetryAfterValue(String v) {
        v = v.trim();
        try {
            return OptionalLong.of(Long.parseLong(v));
        } catch (NumberFormatException e1) {
            try {
                ZonedDateTime when = ZonedDateTime.parse(v, DateTimeFormatter.RFC_1123_DATE_TIME);
                long sec = Duration.between(Instant.now(), when.toInstant()).getSeconds();
                return OptionalLong.of(Math.max(0, sec));
            } catch (DateTimeParseException e2) {
                return OptionalLong.empty();
            }
        }
    }

    private static boolean downloadOne(
            HttpClient client,
            String url,
            String cookieHeader,
            Path outDir,
            String id,
            Duration requestTimeout,
            boolean force) throws InterruptedException, IOException {

        Path part = outDir.resolve(id + ".part");
        Files.deleteIfExists(part);

        int attempt429 = 0;
        int attemptNet = 0;

        while (true) {
            HttpRequest.Builder rb = HttpRequest.newBuilder(URI.create(url))
                    .timeout(requestTimeout)
                    .header("User-Agent", UA)
                    .GET();
            if (!cookieHeader.isEmpty()) {
                rb.header("Cookie", cookieHeader);
            }
            HttpRequest req = rb.build();

            try {
                HttpResponse<Path> resp = client.send(req, HttpResponse.BodyHandlers.ofFile(part,
                        StandardOpenOption.CREATE,
                        StandardOpenOption.TRUNCATE_EXISTING,
                        StandardOpenOption.WRITE));

                int code = resp.statusCode();
                if (code == 429) {
                    Files.deleteIfExists(part);
                    if (attempt429 >= MAX_429_ATTEMPTS) {
                        System.err.println("fail_429_max " + id);
                        return false;
                    }
                    long backoff = (long) (3_000 * Math.pow(2, attempt429) + Math.random() * 1_000);
                    long ra = parseRetryAfterSeconds(resp.headers()).orElse(0) * 1_000L;
                    long wait = Math.max(backoff, ra);
                    System.err.printf(Locale.ROOT, "backoff_429 %s attempt=%d wait_ms=%d%n", id, attempt429 + 1, wait);
                    Thread.sleep(wait);
                    attempt429++;
                    continue;
                }
                if (code >= 500 && code < 600) {
                    Files.deleteIfExists(part);
                    if (attemptNet >= MAX_NETWORK_ATTEMPTS) {
                        System.err.println("fail_5xx_max " + id + " last=" + code);
                        return false;
                    }
                    long wait = (long) (5_000 + Math.random() * 2_000);
                    System.err.printf(Locale.ROOT, "retry_5xx %s code=%d attempt=%d wait_ms=%d%n", id, code, attemptNet + 1, wait);
                    Thread.sleep(wait);
                    attemptNet++;
                    continue;
                }
                if (code != 200) {
                    Files.deleteIfExists(part);
                    System.err.println("fail_http " + id + " " + code);
                    return false;
                }

                String ext = pickExtension(resp.headers(), part);
                Path dest = outDir.resolve(id + "." + ext);
                if (force) {
                    Files.deleteIfExists(dest);
                }
                Files.move(part, dest, StandardCopyOption.REPLACE_EXISTING);
                System.out.println("ok " + id + "." + ext);
                return true;

            } catch (IOException e) {
                Files.deleteIfExists(part);
                if (attemptNet >= MAX_NETWORK_ATTEMPTS) {
                    System.err.println("fail_network_max " + id + " " + e.getMessage());
                    return false;
                }
                long wait = (long) (5_000 + Math.random() * 2_000);
                System.err.printf(Locale.ROOT, "retry_network %s attempt=%d wait_ms=%d msg=%s%n", id, attemptNet + 1, wait, e.getMessage());
                Thread.sleep(wait);
                attemptNet++;
            }
        }
    }

    private static String pickExtension(HttpHeaders headers, Path file) throws IOException {
        String ct = headers.firstValue("Content-Type").orElse("").toLowerCase(Locale.ROOT);
        if (ct.contains("pdf")) {
            return "pdf";
        }
        String cd = headers.firstValue("Content-Disposition").orElse("");
        String fn = parseFilenameFromContentDisposition(cd);
        if (fn != null) {
            String lower = fn.toLowerCase(Locale.ROOT);
            if (lower.endsWith(".pdf")) {
                return "pdf";
            }
        }
        byte[] head = Files.readAllBytes(file);
        if (head.length >= 4 && head[0] == '%' && head[1] == 'P' && head[2] == 'D' && head[3] == 'F') {
            return "pdf";
        }
        return "bin";
    }

    private static String parseFilenameFromContentDisposition(String cd) {
        if (cd == null || cd.isEmpty()) {
            return null;
        }
        int i = cd.toLowerCase(Locale.ROOT).indexOf("filename=");
        if (i < 0) {
            return null;
        }
        String rest = cd.substring(i + "filename=".length()).trim();
        if (rest.startsWith("\"")) {
            int end = rest.indexOf('"', 1);
            if (end > 1) {
                return rest.substring(1, end);
            }
        } else {
            int semi = rest.indexOf(';');
            return semi < 0 ? rest : rest.substring(0, semi).trim();
        }
        return null;
    }
}
