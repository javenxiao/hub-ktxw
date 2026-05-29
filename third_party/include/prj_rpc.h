#ifndef __PRJ_RPC_H__
#define __PRJ_RPC_H__

#include "bb_api.h"

#define PRJ_PSRAM_FILENAME_LEN_MAX   256    /**<上传到PSRAM文件的文件名最大长度*/

#define AT_PARTITION_NAME      "factory1"
#define AT_PARTITION_OFFSET    0
#define AT_PARTITION_SIZE      0x1000

#define AT_DO_PAIR_TIMEOUT_DEFAULT 	12	//unit:s

typedef struct {
    uint8_t     hard_ver;
    uint8_t     app_ver;
 } version_t;

typedef enum {
    PRJ_CMD_SET_ROLE = 0,
    PRJ_CMD_SET_AP_MAC,
    PRJ_CMD_SET_SLOT_MAC,
    PRJ_CMD_SET_BAND,
    PRJ_CMD_SET_RESET_DB,
    PRJ_CMD_SET_PWR,
    PRJ_CMD_SET_USER_DATA,
    PRJ_CMD_SET_PSRAM_START,
    PRJ_CMD_SET_PSRAM_DATA,
    PRJ_CMD_SET_PSRAM_DONE,
    PRJ_CMD_SET_PSRAM_UPDATE,
    PRJ_CMD_SET_MAC,
    PRJ_CMD_SET_WORK_MODE,
    PRJ_CMD_SET_REPEATER_MODE,
    PRJ_CMD_SET_XDS_INTF_EN,
    PRJ_CMD_SET_FREQ_LIST,
    PRJ_CMD_SET_FREQ_POWER_LIMIT,
    PRJ_CMD_SET_PWR_EX,
    PRJ_CMD_GET_BAND = 128,
    PRJ_CMD_GET_ROLE,
    PRJ_CMD_GET_AP_MAC,
    PRJ_CMD_GET_SLOT_MAC,
    PRJ_CMD_GET_PWR,
    PRJ_CMD_GET_RUNSYS,
    PRJ_CMD_GET_MAC,
    PRJ_CMD_GET_WORK_MODE,
    PRJ_CMD_GET_REPEATER_MODE,
    PRJ_CMD_GET_XDS_INTF_EN,
    PRJ_CMD_GET_PRJ_FEATURE,
    PRJ_CMD_GET_FREQ_LIST,
    PRJ_CMD_GET_PWR_EX,
    PRJ_CMD_GET_FREQ_POWER_LIMIT,
    PRJ_CMD_EVENT_DEMO = 200,
    PRJ_CMD_EVENT_PAIR,
    PRJ_CMD_EVENT_PAIR_STOP,
} prj_cmd_e;

PACK(typedef struct {
    uint8_t  cmdid;
    uint16_t length;
    uint8_t  padding[1];
    uint8_t  data[0];
}) prj_rpc_hdr_t;

typedef struct {
    uint8_t role;
} prj_cmd_set_role_t;

typedef struct {
    bb_mac_t ap_mac;
} prj_cmd_set_ap_mac_t;

typedef struct {
    bb_mac_t mac;
} prj_cmd_set_mac_t;

typedef struct {
    uint8_t  slot_id;
    bb_mac_t slot_mac;
} prj_cmd_set_slot_mac_t;

typedef struct {
    uint8_t band_bmp;  // bb_band_e的bitmap
} prj_cmd_set_band_t;
typedef prj_cmd_set_band_t prj_cmd_get_band_t;

typedef struct {
    uint8_t work_mode;  // refers to usr_work_mode_t
} prj_cmd_set_work_mode_t;
typedef prj_cmd_set_work_mode_t prj_cmd_get_work_mode_t;

typedef struct {
    uint8_t repeater_mode;  // 1:relay mode ; else: not relay mode
} prj_cmd_set_repeater_mode_t;
typedef prj_cmd_set_repeater_mode_t prj_cmd_get_repeater_mode_t;

#define prj_cmd_get_ap_mac_t prj_cmd_set_ap_mac_t
#define prj_cmd_get_mac_t prj_cmd_set_mac_t
#define prj_cmd_get_role_t prj_cmd_set_role_t

typedef struct {
    bb_mac_t mac;
}prj_cmd_get_slot_mac_out_t;

typedef struct {
    uint8_t  slot_id;
} prj_cmd_get_slot_mac_in_t;

typedef enum {
    RUNSYS_UNKNOWNN = -1,
    RUNSYS_MASTER,
    RUNSYS_BACKUP
} prj_cmd_runsys_t;

typedef struct {
    prj_cmd_runsys_t runsys_id;
}prj_cmd_get_runsys_out_t;

typedef struct {
    uint8_t  data;
} prj_cmd_event_demo_t;

typedef struct {
    uint8_t bitmap;
    uint8_t padding1[1];
    int16_t timeout;
    uint8_t asyn;
    uint8_t padding2[1];
} prj_cmd_event_pair_t;

/**定义设置命令PRJ_CMD_SET_PSRAM_START的输入参数结构*/
typedef struct {
    uint32_t len;                                               /**<@note img数据的长度*/
    char fname[PRJ_PSRAM_FILENAME_LEN_MAX];                      /**<@note 文件名*/
} prj_cmd_upgrade_psram_start_in_t;

/**定义设置命令PRJ_CMD_SET_PSRAM_DATA的输入参数结构*/
typedef struct {
    uint16_t seq;                                               /**<@note 命令字序列号, 可以任意产生*/
    uint16_t len;                                               /**<@note data中数据的长度*/
    uint8_t data[BB_CFG_PAGE_SIZE-8];                           /**<@note 数据内容*/
} prj_cmd_upgrade_psram_data_in_t;

/**定义设置命令PRJ_CMD_SET_PSRAM_DONE的输入参数结构*/
typedef struct {
    uint32_t crc32;                                             /**<@note crc32校验值*/
} prj_cmd_upgrade_psram_done_in_t;

typedef enum {
    PRJ_PSRAM_ACTION_UPGRADE
} prj_psram_update_action_e;

typedef struct {
    uint16_t action;    //The process of the img, refers to
    uint16_t offset;    //The beginning of the 8030 image
    uint16_t length;    //The length of the 8030 image
}prj_cmd_set_psram_update_t;

typedef struct {
    uint8_t type;                   // Type of the device, refers to xdata_intf_type_t
    uint8_t dev_id;                 // Device id of the interface
    uint8_t slot_id;                // Slot id of the interface, just set 0 if intf is not bb socket port
    uint8_t dir;                    // Dir of the interface, refers to xdata_dir_t
    uint8_t en;                     // Whether enable the interface
} prj_cmd_set_xds_intf_en_t;

typedef struct {
    uint8_t type;                   // Type of the device, refers to xdata_intf_type_t
    uint8_t dev_id;                 // Device id of the interface
    uint8_t slot_id;                // Slot id of the interface, just set 0 if intf is not bb socket port
    uint8_t dir;                    // Dir of the interface, refers to xdata_dir_t
}prj_cmd_get_xds_intf_en_in_t;

typedef struct {
    uint8_t en;                     // Whether enable the interface
} prj_cmd_get_xds_intf_en_out_t;

#define prj_cmd_get_xds_intf_en_t prj_cmd_set_xds_intf_en_t

typedef struct {
    /* rpc config */
    uint8_t usb_rpc_en;
    uint8_t sdio_rpc_en;
    /* xds config */
    uint8_t mblk_pkt_debug;
    uint8_t using_cfg_mem;
    /* bb config */
    uint8_t prj_enable_lna_ctrl;
    /* sys config */
    uint8_t wtd_en;
    uint8_t no_flash;
    uint8_t eth_en;
} prj_cmd_get_prj_feature_out_t;

typedef struct {
    uint32_t freq_num;                                          /**<@note 列表中freq数量*/
    uint32_t freq[BB_CONFIG_MAX_CHAN_NUM];                      /**<@note freq列表*/
} prj_cmd_set_freq_list_t;
typedef prj_cmd_set_freq_list_t prj_cmd_get_freq_list_t;

#if (BB_CONFIG_FREQ_PWR_LIMIT == 1)
typedef struct {
    uint8_t size;                                                   /**<@note 列表大小*/
    bb_freq_pwr_limit_t limit_list[BB_CONFIG_FREQ_POWER_LIMIT_NUM]; /**<@note 频率功率限制列表*/
    uint8_t reserved[3];                                            /**<@note 保留空间*/
} prj_cmd_set_freq_power_limit_t;
typedef prj_cmd_set_freq_power_limit_t prj_cmd_get_freq_power_limit_t;
#endif

typedef struct {
    uint8_t link_auto;              /**<@note follow or otherwise*/
    uint8_t auto_reset;             /**<@note Power after bb reset in auto mode(unit dBm) */
    uint8_t manu_init;              /**<@note Power when bb init in manu mode(unit dBm) */
    uint8_t manu_reset;             /**<@note Power after bb reset in manu mode(unit dBm) */
    uint8_t csma_offset;            /**<@note offset of csma */
}prj_cmd_set_power_ex_t;
typedef prj_cmd_set_power_ex_t prj_cmd_get_power_ex_t;

void prj_rpc_init(void);

void register_hard_ver_callback();
int do_pair(uint8_t bitmap, int16_t timeout);
void do_pair_exit(void);

#endif
