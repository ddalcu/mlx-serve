import { mount } from "svelte";
import App from "./App.svelte";
import { App as Studio } from "./lib/app.svelte";
import "./app.css";

mount(App, { target: document.getElementById("app")!, props: { app: new Studio() } });
