/*
 * firmware/node1_s3/main/main.c
 * ─────────────────────────────────────────────────────────────────────────────
 * NODE 1 — ESP32-S3
 * Router-Reflector Mode: node ping router, thu CSI từ ping reply, gửi UDP.
 *
 * IDF v5.5.x — wifi_csi_config_t dùng wifi_csi_acquire_config_t (nested).
 *
 * Build:
 *   idf.py set-target esp32s3
 *   idf.py flash monitor
 *
 * Packet format gửi về Python:
 *   [0]     node_id    uint8
 *   [1:6]   router_mac 6 bytes — BSSID router
 *   [7]     rssi+128   uint8
 *   [8]     rate       uint8
 *   [9:10]  csi_len    uint16 big-endian
 *   [11:]   csi_data   int8[]
 */

#include <string.h>
#include <stdio.h>
#include <errno.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/event_groups.h"
#include "freertos/queue.h"
#include "esp_wifi.h"
#include "esp_event.h"
#include "esp_netif.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "esp_ping.h"
#include "nvs_flash.h"
#include "lwip/sockets.h"
#include "lwip/inet.h"
#include "lwip/netdb.h"
#include "ping/ping_sock.h"

// ════════════════════════════════════════
// CẤU HÌNH NODE — KHÔNG CẦN ĐỔI CHO NODE 1
// ════════════════════════════════════════
#define NODE_ID           3
#define UDP_PORT          5007
#define SERVER_IP         "192.168.1.168"
#define WIFI_SSID         "PHONG K207"
#define WIFI_PASS         "iotlab123"

// ════════════════════════════════════════
// THAM SỐ HOẠT ĐỘNG
// ════════════════════════════════════════
#define PING_INTERVAL_MS  50        // 20 Hz ping → 20 CSI frames/s
#define CSI_DATA_MAX      256
#define CSI_PKT_MAX_LEN   (CSI_DATA_MAX + 16)
#define CSI_QUEUE_DEPTH   32

// ════════════════════════════════════════
// TYPES
// ════════════════════════════════════════
typedef struct {
    uint8_t  buf[CSI_PKT_MAX_LEN];
    uint16_t len;
} csi_pkt_t;

// ════════════════════════════════════════
// GLOBALS
// ════════════════════════════════════════
static const char        *TAG              = "node1_s3";
static int                g_udp_sock       = -1;
static struct sockaddr_in g_server_addr;
static EventGroupHandle_t g_wifi_evg;
#define WIFI_CONNECTED_BIT BIT0

static QueueHandle_t      g_csi_queue      = NULL;
static uint8_t            g_router_bssid[6]= {0};
static bool               g_bssid_valid    = false;

static volatile uint32_t  g_csi_cb_count   = 0;
static volatile uint32_t  g_csi_enqueued   = 0;
static volatile uint32_t  g_csi_dropped    = 0;


// ════════════════════════════════════════
// WiFi Event Handler
// ════════════════════════════════════════
static void wifi_event_handler(void *arg, esp_event_base_t base,
                               int32_t id, void *data)
{
    if (base == WIFI_EVENT && id == WIFI_EVENT_STA_START) {
        esp_wifi_connect();

    } else if (base == WIFI_EVENT && id == WIFI_EVENT_STA_DISCONNECTED) {
        ESP_LOGW(TAG, "Disconnected, retrying...");
        g_bssid_valid = false;
        xEventGroupClearBits(g_wifi_evg, WIFI_CONNECTED_BIT);
        esp_wifi_connect();

    } else if (base == IP_EVENT && id == IP_EVENT_STA_GOT_IP) {
        ip_event_got_ip_t *ev = (ip_event_got_ip_t *)data;
        ESP_LOGI(TAG, "Node %d IP: " IPSTR, NODE_ID, IP2STR(&ev->ip_info.ip));

        wifi_ap_record_t ap;
        if (esp_wifi_sta_get_ap_info(&ap) == ESP_OK) {
            memcpy(g_router_bssid, ap.bssid, 6);
            g_bssid_valid = true;
            ESP_LOGI(TAG, "Router BSSID: %02x:%02x:%02x:%02x:%02x:%02x",
                     ap.bssid[0], ap.bssid[1], ap.bssid[2],
                     ap.bssid[3], ap.bssid[4], ap.bssid[5]);
        }
        xEventGroupSetBits(g_wifi_evg, WIFI_CONNECTED_BIT);
    }
}

// ════════════════════════════════════════
// WiFi Init
// ════════════════════════════════════════
static void wifi_init_sta(void)
{
    g_wifi_evg = xEventGroupCreate();
    esp_netif_create_default_wifi_sta();

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));

    ESP_ERROR_CHECK(esp_event_handler_register(
        WIFI_EVENT, ESP_EVENT_ANY_ID, wifi_event_handler, NULL));
    ESP_ERROR_CHECK(esp_event_handler_register(
        IP_EVENT, IP_EVENT_STA_GOT_IP, wifi_event_handler, NULL));

    wifi_config_t wc = {};
    strlcpy((char *)wc.sta.ssid,     WIFI_SSID, sizeof(wc.sta.ssid));
    strlcpy((char *)wc.sta.password, WIFI_PASS, sizeof(wc.sta.password));
    wc.sta.listen_interval = 1;

    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &wc));
    ESP_ERROR_CHECK(esp_wifi_start());

    xEventGroupWaitBits(g_wifi_evg, WIFI_CONNECTED_BIT,
                        pdFALSE, pdFALSE, pdMS_TO_TICKS(15000));
}

// ════════════════════════════════════════
// UDP Socket Init
// ════════════════════════════════════════
static void udp_init(void)
{
    g_udp_sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (g_udp_sock < 0) {
        ESP_LOGE(TAG, "Cannot create socket: errno %d", errno);
        return;
    }
    memset(&g_server_addr, 0, sizeof(g_server_addr));
    g_server_addr.sin_family = AF_INET;
    g_server_addr.sin_port   = htons(UDP_PORT);
    inet_aton(SERVER_IP, &g_server_addr.sin_addr);
    ESP_LOGI(TAG, "UDP -> %s:%d", SERVER_IP, UDP_PORT);
}

// ════════════════════════════════════════
// CSI Callback (ESP32-S3, IDF v5.5.x)
// wifi_csi_acquire_config_t — KHÔNG có IRAM_ATTR vì gọi từ task context
// ════════════════════════════════════════
static void wifi_csi_rx_cb(void *ctx, wifi_csi_info_t *info)
{
    g_csi_cb_count++;

    if (!g_csi_queue || !info || info->len == 0) return;
    if (!g_bssid_valid) return;
    if (memcmp(info->mac, g_router_bssid, 6) != 0) return;

    csi_pkt_t pkt;
    int pos = 0;

    pkt.buf[pos++] = (uint8_t)NODE_ID;
    memcpy(pkt.buf + pos, info->mac, 6); pos += 6;
    pkt.buf[pos++] = (uint8_t)((int)info->rx_ctrl.rssi + 128);
    pkt.buf[pos++] = (uint8_t)info->rx_ctrl.rate;

    uint16_t clen = (info->len > CSI_DATA_MAX) ? CSI_DATA_MAX : (uint16_t)info->len;
    pkt.buf[pos++] = (uint8_t)(clen >> 8);
    pkt.buf[pos++] = (uint8_t)(clen & 0xFF);
    memcpy(pkt.buf + pos, info->buf, clen); pos += clen;
    pkt.len = (uint16_t)pos;

    if (xQueueSend(g_csi_queue, &pkt, 0) == pdTRUE) {
        g_csi_enqueued++;
    } else {
        g_csi_dropped++;
    }
}

// ════════════════════════════════════════
// CSI Init — ESP32-S3 / IDF v5.5.x
//
// ESP32-S3 vẫn dùng wifi_csi_config_t với các field CŨ (lltf_en, htltf_en...)
// Nhưng IDF v5.5 đổi tên struct thành wifi_csi_acquire_config_t tuỳ chip.
// Kiểm tra: nếu build lỗi "no member named lltf_en", dùng nhánh #else bên dưới.
// ════════════════════════════════════════
static void csi_init(void)
{
    // ESP32-S3 + IDF v5.5.1: wifi_csi_config_t vẫn dùng field phẳng CŨ
    // (field .shift đã bị xoá từ IDF v5, nhưng các field còn lại vẫn giữ)
    wifi_csi_config_t cfg = {
        .lltf_en           = true,   // Legacy Long Training Field — ổn định nhất
        .htltf_en          = false,  // Không cần HT-LTF với router thông thường
        .stbc_htltf2_en    = false,
        .ltf_merge_en      = true,
        .channel_filter_en = true,
        .manu_scale        = true,
    };

    ESP_ERROR_CHECK(esp_wifi_set_csi_config(&cfg));
    ESP_ERROR_CHECK(esp_wifi_set_csi_rx_cb(wifi_csi_rx_cb, NULL));
    ESP_ERROR_CHECK(esp_wifi_set_csi(true));
    ESP_LOGI(TAG, "CSI enabled — lltf_en=1, channel_filter=1, manu_scale=1");
}

// ════════════════════════════════════════
// UDP Send Task
// ════════════════════════════════════════
static void udp_send_task(void *arg)
{
    xEventGroupWaitBits(g_wifi_evg, WIFI_CONNECTED_BIT,
                        pdFALSE, pdFALSE, portMAX_DELAY);
    ESP_LOGI(TAG, "udp_send_task running");

    csi_pkt_t pkt;
    uint32_t  ok = 0, fail = 0, last_log = 0;

    while (1) {
        if (xQueueReceive(g_csi_queue, &pkt, pdMS_TO_TICKS(500)) != pdTRUE)
            continue;
        if (g_udp_sock < 0) continue;

        int r = sendto(g_udp_sock, pkt.buf, pkt.len, 0,
                       (struct sockaddr *)&g_server_addr, sizeof(g_server_addr));
        if (r > 0) { ok++; }
        else {
            fail++;
            if (errno == ENOMEM) vTaskDelay(pdMS_TO_TICKS(5));
        }

        uint32_t now = (uint32_t)(esp_timer_get_time() / 1000);
        if (now - last_log >= 5000) {
            last_log = now;
            ESP_LOGI(TAG,
                "[STAT] cb=%lu enq=%lu drop=%lu udp_ok=%lu udp_fail=%lu",
                (unsigned long)g_csi_cb_count,
                (unsigned long)g_csi_enqueued,
                (unsigned long)g_csi_dropped,
                (unsigned long)ok,
                (unsigned long)fail);
        }
    }
}

// ════════════════════════════════════════
// Ping Task — dùng esp_ping (IDF component, ổn định hơn raw ICMP)
// ════════════════════════════════════════
static void ping_task(void *arg)
{
    xEventGroupWaitBits(g_wifi_evg, WIFI_CONNECTED_BIT,
                        pdFALSE, pdFALSE, portMAX_DELAY);

    // Lấy IP gateway từ DHCP
    esp_netif_t *netif = esp_netif_get_handle_from_ifkey("WIFI_STA_DEF");
    if (!netif) {
        ESP_LOGE(TAG, "Cannot get netif handle");
        vTaskDelete(NULL);
        return;
    }

    esp_netif_ip_info_t ip_info;
    ESP_ERROR_CHECK(esp_netif_get_ip_info(netif, &ip_info));

    char gw_str[16];
    esp_ip4addr_ntoa(&ip_info.gw, gw_str, sizeof(gw_str));
    ESP_LOGI(TAG, "Pinging gateway: %s every %d ms", gw_str, PING_INTERVAL_MS);

    // Tạo raw ICMP socket — nhẹ hơn esp_ping component
    int sock = socket(AF_INET, SOCK_RAW, IPPROTO_ICMP);
    if (sock < 0) {
        ESP_LOGE(TAG, "ICMP socket fail: errno=%d", errno);
        vTaskDelete(NULL);
        return;
    }

    struct timeval tv = { .tv_sec = 1, .tv_usec = 0 };
    setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

    struct sockaddr_in dest = {
        .sin_family      = AF_INET,
        .sin_addr.s_addr = ip_info.gw.addr,
    };

    // ICMP Echo packet
    typedef struct __attribute__((packed)) {
        uint8_t  type, code;
        uint16_t chksum, id, seq;
        uint8_t  data[1];
    } icmp_echo_t;

    uint8_t buf[sizeof(icmp_echo_t)];
    icmp_echo_t *icmp = (icmp_echo_t *)buf;
    uint16_t seq = 0;

    while (1) {
        memset(icmp, 0, sizeof(buf));
        icmp->type = 8;  // ECHO REQUEST
        icmp->id   = htons((uint16_t)NODE_ID);
        icmp->seq  = htons(seq++);

        // Tính checksum
        uint32_t sum = 0;
        uint16_t *ptr = (uint16_t *)buf;
        for (int i = 0; i < (int)(sizeof(buf) / 2); i++) sum += ptr[i];
        while (sum >> 16) sum = (sum & 0xffff) + (sum >> 16);
        icmp->chksum = (uint16_t)(~sum);

        sendto(sock, buf, sizeof(buf), 0,
               (struct sockaddr *)&dest, sizeof(dest));

        vTaskDelay(pdMS_TO_TICKS(PING_INTERVAL_MS));
    }
}

// ════════════════════════════════════════
// app_main
// ════════════════════════════════════════
void app_main(void)
{
    ESP_LOGI(TAG, "=== CSI Node %d (ESP32-S3, IDF v5.5.x) ===", NODE_ID);

    // NVS
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES ||
        ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);

    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());

    // CSI queue
    g_csi_queue = xQueueCreate(CSI_QUEUE_DEPTH, sizeof(csi_pkt_t));
    if (!g_csi_queue) {
        ESP_LOGE(TAG, "Queue create failed!");
        return;
    }

    wifi_init_sta();
    xEventGroupWaitBits(g_wifi_evg, WIFI_CONNECTED_BIT,
                        pdFALSE, pdFALSE, portMAX_DELAY);

    udp_init();
    csi_init();

    xTaskCreate(udp_send_task, "udp_tx",   4096, NULL, 6, NULL);
    xTaskCreate(ping_task,     "ping",     4096, NULL, 5, NULL);

    ESP_LOGI(TAG, "Node %d ready — %d Hz ping → CSI → UDP %s:%d",
             NODE_ID, 1000 / PING_INTERVAL_MS, SERVER_IP, UDP_PORT);
}