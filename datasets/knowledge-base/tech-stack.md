# 我擅长的技术栈

## 来源与用法

- 抽象自主力工程 `dmp-mastar`（`dmp-web` / `iot-v3` / `dev-res-vue-admin` / `dmp-res-vue-cmp` / `go-dmp-util`）。
- 这些是我熟悉、默认可高质量交付的技术；选型、实现与代码评审时优先按此栈给出方案。
- 项目已有技术选择优先于本文；仅在「新建能力 / 可自由选型 / 用户未指定」时用本文作默认。

## 领域画像

- **B 端 / 运营端物联网与设备管理平台（DMP）**：多租户企业端、运营管理端、设备/媒体/OTA/开放 API、计费积分等。
- **IoT 接入与实时通道**：网关、WebSocket / Netty 长连接、资源与门禁类 API。
- **数据与运维工具**：Go CLI 做导入导出、环境同步、第三方开放平台联调、ASR/运营灌数等。

## 后端（主栈）

| 层级 | 技术 | 备注 |
|------|------|------|
| 语言 | Java 8 | 主力业务与 IoT 服务 |
| 框架 | Spring Boot 2.x | `dmp-web` ~2.3/2.7 混用；`iot-v3` ~2.7 |
| 微服务 | Spring Cloud + Spring Cloud Alibaba | Nacos 配置/发现、OpenFeign、LoadBalancer |
| 网关/服务拆分 | 多模块 Maven（`*-api` / `*-admin` / `*-cmp` / `*-mp` / `*-base` / `common`） | 按端与领域拆服务 |
| ORM | MyBatis-Plus | Mapper XML 可放在 `src/main/java` |
| 数据库 | MySQL + Druid | |
| 文档库 | MongoDB | 主要用于 `iot-v3` |
| 缓存/锁 | Redis + Redisson | |
| 消息 | RocketMQ | 部分业务服务 |
| 安全 | Spring Security | |
| API 文档 | Springfox / Swagger、Knife4j | |
| 可观测 | Sleuth / Zipkin | |
| 序列化/工具 | Jackson、Gson、Fastjson2、Lombok、EasyExcel | |
| 部署 | Docker 多环境（dev / prod / intl）+ Nacos 外置配置 | |

实现习惯：

- 沿用现有多模块边界与 `common` / `base` 复用，不随意新开仓库或平行栈。
- 配置走 Nacos；本地/容器通过 `spring.cloud.nacos.*` 与 profile 区分环境。
- 改接口时同步考虑管理端、B 端（CMP）、开放 API 等多端调用方。
- **运行时是 Java 8**（测试容器常见 OpenJDK 8）。本机 JDK 更高且 Maven 只有 `source/target=8` 时，仍可能编过 Java 9+ API，上线 `NoSuchMethodError`。只使用 Java 8 就有的方法；详见 `memory/corrections.md`。

## IoT / 实时

- Spring WebSocket、Netty（`iot-v3`：`ws-hub` / `ws-bus` / `gate-api` / `res-api` 等）。
- 长连接、网关与资源 API 分离；与主站 `dmp-web` 通过 Feign / HTTP 协作。

## 前端（主栈）

| 层级 | 技术 | 备注 |
|------|------|------|
| 框架 | Vue 3 | Composition API / SFC |
| 语言 | JavaScript（ESM） | 非 TypeScript 为主 |
| 构建 | Vite 5 | `dev` / `test` / `prod` / `intl` 多 mode |
| UI | Element Plus | |
| 状态/路由 | Pinia、Vue Router 4 | |
| HTTP | Axios | |
| 图表/工具 | ECharts、lodash、file-saver、bignumber.js 等 | |

端分工（术语）：

- **管理端 / 运营端** → `dev-res-vue-admin`
- **B 端 / CMP / 企业端** → `dmp-res-vue-cmp`
- 未指明端时不要混改两个前端；用户说「B 端」只动 CMP。

## Go（辅栈 / 工具）

- Go 1.22，`cmd/` + `internal/` + `configs/` 布局。
- 用途：批处理、导入导出、Excel（excelize）、YAML 配置、HTTP 客户端联调、运维脚本替代品。
- 可直接跑 `cmd/*/main.go`；敏感配置本地文件，不入库。

## 脚本与工程化

- Python 3：仓库内校验/统计类脚本（如 i18n check、周报收集）。
- Maven 多模块、Makefile、分环境 Docker 构建脚本。
- Cursor / Codex：项目内 `.cursor/rules` 与 skills 承载工作流；跨项目共性进本知识库。

## 默认选型倾向

当用户未指定技术时：

1. **业务服务** → Java + Spring Boot + Spring Cloud Alibaba + MyBatis-Plus + MySQL + Redis。
2. **管理/B 端页面** → Vue 3 + Vite + Element Plus + Pinia（JS，不默认上 TS）。
3. **一次性数据/联调工具 / DMP 脚本** → `go-dmp-util`（用户说「加 DMP 脚本」即指此仓，见 `projects/go-dmp-util.md`）。
4. **实时设备通道** → 参考 `iot-v3`（WebSocket / Netty），不轻易换新协议栈。
5. **避免默认推荐**：Nest/Next 全栈、Kotlin、React、TypeScript 强制、JPA/Hibernate、无必要的新中间件——除非用户明确要求或项目已在用。

## 非擅长 / 需谨慎引入

- 非本栈的前端框架（React、Angular、Svelte）与强制 TypeScript 改造。
- 与现网差异大的 Java 版本跃迁（如直接上 Boot 3 / Jakarta）——需单独评估兼容性。
- 为炫技引入的新消息队列、新 ORM、新网关，替代已稳定的 Nacos / Feign / RocketMQ / MyBatis-Plus。
