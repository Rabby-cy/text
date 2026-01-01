using System;
using System.Collections.Generic;
using System.Linq;
using System.Net;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;
using Verse;
using UnityEngine;
using RimWorld;

namespace RimTalk.Memory.AI
{
    // ? v3.3.21: Response DTO classes for robust JSON parsing
    [Serializable]
    public class OpenAIResponse
    {
        public Choice[] choices;
    }
    
    [Serializable]
    public class Choice
    {
        public Message message;
    }
    
    [Serializable]
    public class Message
    {
        public string content;
    }
    
    [Serializable]
    public class GeminiResponse
    {
        public Candidate[] candidates;
    }
    
    [Serializable]
    public class Candidate
    {
        public Content content;
    }
    
    [Serializable]
    public class Content
    {
        public Part[] parts;
    }
    
    [Serializable]
    public class Part
    {
        public string text;
    }
    
    // ? Request DTO classes
    [Serializable]
    public class OpenAIRequest
    {
        public string model;
        public OpenAIMessage[] messages;
        public float temperature;
        public int max_tokens;
        public bool enable_prompt_cache; // DeepSeek
    }
    
    [Serializable]
    public class OpenAIMessage
    {
        public string role;
        public string content;
        public CacheControl cache_control; // OpenAI Prompt Caching
        public bool cache; // DeepSeek cache
    }
    
    [Serializable]
    public class CacheControl
    {
        public string type;
    }
    
    [Serializable]
    public class GeminiRequest
    {
        public GeminiContent[] contents;
        public GeminiGenerationConfig generationConfig;
    }
    
    [Serializable]
    public class GeminiContent
    {
        public GeminiPart[] parts;
    }
    
    [Serializable]
    public class GeminiPart
    {
        public string text;
    }
    
    [Serializable]
    public class GeminiGenerationConfig
    {
        public float temperature;
        public int maxOutputTokens;
        public GeminiThinkingConfig thinkingConfig;
    }
    
    [Serializable]
    public class GeminiThinkingConfig
    {
        public int thinkingBudget;
    }
    
    public static class IndependentAISummarizer
    {
        private static bool isInitialized = false;
        private static string apiKey, apiUrl, model, provider;
        
        private const int MAX_CACHE_SIZE = 100;
        private const int CACHE_CLEANUP_THRESHOLD = 120;
        
        private static readonly Dictionary<string, string> completedSummaries = new Dictionary<string, string>();
        private static readonly HashSet<string> pendingSummaries = new HashSet<string>();
        private static readonly Dictionary<string, List<Action<string>>> callbackMap = new Dictionary<string, List<Action<string>>>();
        private static readonly Queue<Action> mainThreadActions = new Queue<Action>();

        public static string ComputeCacheKey(Pawn pawn, List<MemoryEntry> memories)
        {
            var ids = memories.Select(m => m.id ?? m.content.GetHashCode().ToString()).ToArray();
            string joinedIds = string.Join("|", ids);
            return $"{pawn.ThingID}_{memories.Count}_{joinedIds.GetHashCode()}";
        }

        public static void RegisterCallback(string cacheKey, Action<string> callback)
        {
            lock (callbackMap)
            {
                if (!callbackMap.TryGetValue(cacheKey, out var callbacks))
                {
                    callbacks = new List<Action<string>>();
                    callbackMap[cacheKey] = callbacks;
                }
                callbacks.Add(callback);
            }
        }

        public static void ProcessPendingCallbacks(int maxPerTick = 5)
        {
            int processed = 0;
            lock (mainThreadActions)
            {
                while (mainThreadActions.Count > 0 && processed < maxPerTick)
                {
                    try
                    {
                        mainThreadActions.Dequeue()?.Invoke();
                    }
                    catch (Exception ex)
                    {
                        Log.Error($"[AI Summarizer] Callback error: {ex.Message}");
                    }
                    processed++;
                }
            }
        }

        public static void ForceReinitialize()
        {
            isInitialized = false;
            Initialize();
        }
        
        public static void ClearAllConfiguration()
        {
            apiKey = "";
            apiUrl = "";
            model = "";
            provider = "";
            isInitialized = false;
            
            lock (completedSummaries)
            {
                completedSummaries.Clear();
            }
            
            lock (pendingSummaries)
            {
                pendingSummaries.Clear();
            }
            
            lock (callbackMap)
            {
                callbackMap.Clear();
            }
            
            lock (mainThreadActions)
            {
                mainThreadActions.Clear();
            }
            
            Log.Message("[AI] ? All API configuration and cache cleared");
        }
        
        public static void Initialize()
        {
            try
            {
                var settings = RimTalk.MemoryPatch.RimTalkMemoryPatchMod.Settings;
                
                if (settings.useRimTalkAIConfig)
                {
                    if (TryLoadFromRimTalk())
                    {
                        if (ValidateConfiguration())
                        {
                            Log.Message($"[AI] ? Loaded from RimTalk ({provider}/{model})");
                            isInitialized = true;
                            return;
                        }
                        else
                        {
                            Log.Warning("[AI] ? RimTalk config invalid, using independent config");
                        }
                    }
                    else
                    {
                        Log.Warning("[AI] ? RimTalk not configured, using independent config as fallback");
                    }
                }
                
                apiKey = settings.independentApiKey;
                apiUrl = settings.independentApiUrl;
                model = settings.independentModel;
                provider = settings.independentProvider;
                
                if (provider == "Player2")
                {
                    if (isPlayer2Local && !string.IsNullOrEmpty(player2LocalKey))
                    {
                        apiKey = player2LocalKey;
                        apiUrl = $"{Player2LocalUrl}/chat/completions";
                        Log.Message("[AI] ? Using Player2 local app connection");
                    }
                    else if (!string.IsNullOrEmpty(apiKey))
                    {
                        apiUrl = $"{Player2RemoteUrl}/chat/completions";
                        Log.Message("[AI] ? Using Player2 remote API with manual key");
                    }
                    else
                    {
                        Log.Message("[AI] ? Player2 selected but no key, trying to detect local app...");
                        TryDetectPlayer2LocalApp();
                    }
                }
                
                if (string.IsNullOrEmpty(apiUrl))
                {
                    if (provider == "OpenAI")
                    {
                        apiUrl = "https://api.openai.com/v1/chat/completions";
                    }
                    else if (provider == "DeepSeek")
                    {
                        apiUrl = "https://api.deepseek.com/v1/chat/completions";
                    }
                    else if (provider == "Google")
                    {
                        apiUrl = "https://generativelanguage.googleapis.com/v1beta/models/MODEL_PLACEHOLDER:generateContent?key=API_KEY_PLACEHOLDER";
                    }
                    else if (provider == "Player2")
                    {
                        apiUrl = $"{Player2RemoteUrl}/chat/completions";
                    }
                }
                
                if (!ValidateConfiguration())
                {
                    isInitialized = false;
                    return;
                }
                
                Log.Message($"[AI] ? Initialized with independent config ({provider}/{model})");
                Log.Message($"[AI]    API Key: {SanitizeApiKey(apiKey)}");
                Log.Message($"[AI]    API URL: {apiUrl}");
                isInitialized = true;
            }
            catch (Exception ex)
            {
                Log.Error($"[AI] ? Init failed: {ex.Message}");
                isInitialized = false;
            }
        }
        
        private static bool ValidateConfiguration()
        {
            if (string.IsNullOrEmpty(apiKey))
            {
                Log.Error("[AI] ? API Key is empty!");
                Log.Error("[AI]    Please configure in: Options → Mod Settings → RimTalk-Expand Memory → AI配置");
                return false;
            }
            
            if (apiKey.Length < 10)
            {
                Log.Error($"[AI] ? API Key too short (length: {apiKey.Length})!");
                Log.Error("[AI]    Valid API Keys are usually 20+ characters");
                Log.Error($"[AI]    Your key: {SanitizeApiKey(apiKey)}");
                return false;
            }
            
            if (provider != "Custom" && provider != "Player2" && provider != "Google")
            {
                if ((provider == "OpenAI" || provider == "DeepSeek") && !apiKey.StartsWith("sk-"))
                {
                    Log.Warning($"[AI] ? API Key doesn't start with 'sk-' for {provider}");
                    Log.Warning($"[AI]    Your key: {SanitizeApiKey(apiKey)}");
                    Log.Warning("[AI]    If using third-party proxy, select 'Custom' or 'Player2' provider");
                }
            }
            
            if (string.IsNullOrEmpty(apiUrl))
            {
                Log.Error("[AI] ? API URL is empty!");
                return false;
            }
            
            if (string.IsNullOrEmpty(model))
            {
                Log.Warning("[AI] ? Model name is empty, using default");
                model = "gpt-3.5-turbo";
            }
            
            return true;
        }
        
        private static string SanitizeApiKey(string key)
        {
            if (string.IsNullOrEmpty(key))
                return "(empty)";
            
            if (key.Length <= 10)
                return key.Substring(0, Math.Min(3, key.Length)) + "...";
            
            return $"{key.Substring(0, 7)}...{key.Substring(key.Length - 4)} (length: {key.Length})";
        }
        
        private static bool TryLoadFromRimTalk()
        {
            try
            {
                Assembly assembly = AppDomain.CurrentDomain.GetAssemblies().FirstOrDefault((Assembly a) => a.GetName().Name == "RimTalk");
                if (assembly == null) return false;
                
                Type type = assembly.GetType("RimTalk.Settings");
                if (type == null) return false;
                
                MethodInfo method = type.GetMethod("Get", BindingFlags.Static | BindingFlags.Public);
                if (method == null) return false;
                
                object obj = method.Invoke(null, null);
                if (obj == null) return false;
                
                Type type2 = obj.GetType();
                MethodInfo method2 = type2.GetMethod("GetActiveConfig");
                if (method2 == null) return false;
                
                object obj2 = method2.Invoke(obj, null);
                if (obj2 == null) return false;
                
                Type type3 = obj2.GetType();
                
                FieldInfo field = type3.GetField("ApiKey");
                if (field != null)
                {
                    apiKey = (field.GetValue(obj2) as string);
                }
                
                FieldInfo field2 = type3.GetField("BaseUrl");
                if (field2 != null)
                {
                    apiUrl = (field2.GetValue(obj2) as string);
                }
                
                if (string.IsNullOrEmpty(apiUrl))
                {
                    FieldInfo field3 = type3.GetField("Provider");
                    if (field3 != null)
                    {
                        object value = field3.GetValue(obj2);
                        provider = value.ToString();
                        
                        if (provider == "OpenAI")
                        {
                            apiUrl = "https://api.openai.com/v1/chat/completions";
                        }
                        else if (provider == "DeepSeek")
                        {
                            apiUrl = "https://api.deepseek.com/v1/chat/completions";
                        }
                        else if (provider == "Google")
                        {
                            apiUrl = "https://generativelanguage.googleapis.com/v1beta/models/MODEL_PLACEHOLDER:generateContent?key=API_KEY_PLACEHOLDER";
                        }
                        else if (provider == "Player2")
                        {
                            apiUrl = "https://api.player2.live/v1/chat/completions";
                        }
                    }
                }
                
                FieldInfo field4 = type3.GetField("SelectedModel");
                if (field4 != null)
                {
                    model = (field4.GetValue(obj2) as string);
                }
                else
                {
                    FieldInfo field5 = type3.GetField("CustomModelName");
                    if (field5 != null)
                    {
                        model = (field5.GetValue(obj2) as string);
                    }
                }
                
                if (string.IsNullOrEmpty(model))
                {
                    model = "gpt-3.5-turbo";
                }
                
                if (!string.IsNullOrEmpty(apiKey) && !string.IsNullOrEmpty(apiUrl))
                {
                    Log.Message($"[AI] Loaded from RimTalk ({provider}/{model})");
                    isInitialized = true;
                    return true;
                }
                
                return false;
            }
            catch (Exception)
            {
                return false;
            }
        }

        public static bool IsAvailable()
        {
            if (!isInitialized) Initialize();
            return isInitialized;
        }

        public static string SummarizeMemories(Pawn pawn, List<MemoryEntry> memories, string promptTemplate)
        {
            if (!IsAvailable()) return null;

            string cacheKey = ComputeCacheKey(pawn, memories);

            lock (completedSummaries)
            {
                if (completedSummaries.TryGetValue(cacheKey, out string summary))
                {
                    return summary;
                }
            }

            lock (pendingSummaries)
            {
                if (pendingSummaries.Contains(cacheKey)) return null;
                pendingSummaries.Add(cacheKey);
            }

            string prompt = BuildPrompt(pawn, memories, promptTemplate);

            Task.Run(async () =>
            {
                try
                {
                    string result = await CallAIAsync(prompt);
                    if (result != null)
                    {
                        lock (completedSummaries)
                        {
                            if (completedSummaries.Count >= CACHE_CLEANUP_THRESHOLD)
                            {
                                var toRemove = completedSummaries.Keys
                                    .OrderBy(k => k, StringComparer.Ordinal)
                                    .Take(MAX_CACHE_SIZE / 2)
                                    .ToList();
                                
                                foreach (var key in toRemove)
                                {
                                    completedSummaries.Remove(key);
                                }
                                
                                if (Prefs.DevMode)
                                {
                                    Log.Message($"[AI Summarizer] ? Cleaned cache: {toRemove.Count} entries removed, {completedSummaries.Count} remaining");
                                }
                            }
                            
                            completedSummaries[cacheKey] = result;
                        }
                        lock (callbackMap)
                        {
                            if (callbackMap.TryGetValue(cacheKey, out var callbacks))
                            {
                                foreach (var cb in callbacks)
                                {
                                    lock (mainThreadActions)
                                    {
                                        mainThreadActions.Enqueue(() => cb(result));
                                    }
                                }
                                callbackMap.Remove(cacheKey);
                            }
                        }
                    }
                }
                catch (Exception ex)
                {
                    Log.Error($"[AI Summarizer] Task failed: {ex.Message}");
                }
                finally
                {
                    lock (pendingSummaries)
                    {
                        pendingSummaries.Remove(cacheKey);
                    }
                }
            });

            return null;
        }

        private static string BuildPrompt(Pawn pawn, List<MemoryEntry> memories, string template)
        {
            var sb = new StringBuilder();
            
            if (template == "deep_archive")
            {
                sb.AppendLine($"请为殖民者{pawn.LabelShort}创建深度归档");
                sb.AppendLine();
                sb.AppendLine("记忆列表：");
                int i = 1;
                foreach (var m in memories.Take(15))
                {
                    sb.AppendLine($"{i}. {m.content}");
                    i++;
                }
                sb.AppendLine();
                sb.AppendLine("要求：保留关键事件和人物关系记录");
                sb.AppendLine("合并相似经历，突出重要转折");
                sb.AppendLine("控制在不超过60字");
                sb.AppendLine("只输出总结内容，不要解释或格式");
            }
            else
            {
                sb.AppendLine($"请为殖民者{pawn.LabelShort}创建记忆总结");
                sb.AppendLine();
                sb.AppendLine("记忆列表：");
                int i = 1;
                foreach (var m in memories.Take(20))
                {
                    sb.AppendLine($"{i}. {m.content}");
                    i++;
                }
                sb.AppendLine();
                sb.AppendLine("要求：聚焦主要活动和事件");
                sb.AppendLine("相同事件合并，注明频次");
                sb.AppendLine("控制在不超过80字");
                sb.AppendLine("只输出总结内容，不要解释或格式");
            }
            
            return sb.ToString();
        }

        private static string BuildJsonRequest(string prompt)
        {
            bool isGoogle = (provider == "Google");
            var settings = RimTalk.MemoryPatch.RimTalkMemoryPatchMod.Settings;
            bool enableCaching = settings != null && settings.enablePromptCaching;
            
            if (isGoogle)
            {
                string escapedPrompt = EscapeJsonString(prompt);
                
                var sb = new StringBuilder();
                sb.Append("{");
                sb.Append("\"contents\":[{");
                sb.Append("\"parts\":[{");
                sb.Append($"\"text\":\"{escapedPrompt}\"");
                sb.Append("}]");
                sb.Append("}],");
                sb.Append("\"generationConfig\":{");
                sb.Append("\"temperature\":0.7,");
                sb.Append("\"maxOutputTokens\":200");
                
                if (model.Contains("flash"))
                {
                    sb.Append(",\"thinkingConfig\":{\"thinkingBudget\":0}");
                }
                
                sb.Append("}");
                sb.Append("}");
                
                return sb.ToString();
            }
            else
            {
                string systemPrompt = "你是一个RimWorld殖民者的记忆总结助手。\n" +
                                    "请简洁准确地总结记忆内容。\n" +
                                    "只输出总结内容，不要解释或格式。";
                
                string escapedSystem = EscapeJsonString(systemPrompt);
                string escapedPrompt = EscapeJsonString(prompt);
                
                var sb = new StringBuilder();
                sb.Append("{");
                sb.Append($"\"model\":\"{model}\",");
                sb.Append("\"messages\":[");
                
                sb.Append("{\"role\":\"system\",");
                sb.Append($"\"content\":\"{escapedSystem}\"");
                
                if (enableCaching)
                {
                    if ((provider == "OpenAI" || provider == "Custom" || provider == "Player2") && 
                        (model.Contains("gpt-4") || model.Contains("gpt-3.5")))
                    {
                        sb.Append(",\"cache_control\":{\"type\":\"ephemeral\"}");
                    }
                    else if (provider == "DeepSeek")
                    {
                        sb.Append(",\"cache\":true");
                    }
                }
                
                sb.Append("},");
                
                sb.Append("{\"role\":\"user\",");
                sb.Append($"\"content\":\"{escapedPrompt}\"");
                sb.Append("}],");
                
                sb.Append("\"temperature\":0.7,");
                sb.Append("\"max_tokens\":200");
                
                if (enableCaching && provider == "DeepSeek")
                {
                    sb.Append(",\"enable_prompt_cache\":true");
                }
                
                sb.Append("}");
                
                return sb.ToString();
            }
        }
        
        private static string EscapeJsonString(string text)
        {
            if (string.IsNullOrEmpty(text))
                return "";
            
            var sb = new StringBuilder(text.Length + 20);
            
            foreach (char c in text)
            {
                switch (c)
                {
                    case '\"':
                        sb.Append("\\\"");
                        break;
                    case '\\':
                        sb.Append("\\\\");
                        break;
                    case '\n':
                        sb.Append("\\n");
                        break;
                    case '\r':
                        sb.Append("\\r");
                        break;
                    case '\t':
                        sb.Append("\\t");
                        break;
                    case '\b':
                        sb.Append("\\b");
                        break;
                    case '\f':
                        sb.Append("\\f");
                        break;
                    default:
                        if (c < 32)
                        {
                            sb.Append("\\u");
                            sb.Append(((int)c).ToString("x4"));
                        }
                        else
                        {
                            sb.Append(c);
                        }
                        break;
                }
            }
            
            return sb.ToString();
        }

        /// <summary>
        /// ? v3.3.21: Optimized timeout to 60 seconds (was 120)
        /// </summary>
        private static async Task<string> CallAIAsync(string prompt)
        {
            const int MAX_RETRIES = 3;
            const int RETRY_DELAY_MS = 2000;
            
            for (int attempt = 1; attempt <= MAX_RETRIES; attempt++)
            {
                try
                {
                    string actualUrl = apiUrl;
                    if (provider == "Google")
                    {
                        actualUrl = apiUrl.Replace("MODEL_PLACEHOLDER", model).Replace("API_KEY_PLACEHOLDER", apiKey);
                    }

                    if (attempt > 1)
                    {
                        Log.Message($"[AI Summarizer] Retry attempt {attempt}/{MAX_RETRIES}...");
                    }
                    else
                    {
                        Log.Message($"[AI Summarizer] Calling API: {actualUrl.Substring(0, Math.Min(60, actualUrl.Length))}...");
                        Log.Message($"[AI Summarizer]   Provider: {provider}");
                        Log.Message($"[AI Summarizer]   Model: {model}");
                        Log.Message($"[AI Summarizer]   API Key: {SanitizeApiKey(apiKey)}");
                    }

                    var request = (HttpWebRequest)WebRequest.Create(actualUrl);
                    request.Method = "POST";
                    request.ContentType = "application/json";
                    
                    if (provider != "Google")
                    {
                        request.Headers["Authorization"] = $"Bearer {apiKey}";
                    }
                    
                    // ? OPTIMIZED: Reduced timeout from 120s to 60s
                    request.Timeout = 60000; // 60 seconds

                    string json = BuildJsonRequest(prompt);
                    byte[] bodyRaw = Encoding.UTF8.GetBytes(json);
                    request.ContentLength = bodyRaw.Length;

                    using (var stream = await request.GetRequestStreamAsync())
                    {
                        await stream.WriteAsync(bodyRaw, 0, bodyRaw.Length);
                    }

                    using (var response = (HttpWebResponse)await request.GetResponseAsync())
                    using (var streamReader = new System.IO.StreamReader(response.GetResponseStream()))
                    {
                        string responseText = await streamReader.ReadToEndAsync();
                        string result = ParseJsonResponse(responseText);
                        
                        if (attempt > 1)
                        {
                            Log.Message($"[AI Summarizer] ? Retry successful on attempt {attempt}");
                        }
                        
                        return result;
                    }
                }
                catch (WebException ex)
                {
                    bool shouldRetry = false;
                    string errorDetail = "";
                    HttpStatusCode statusCode = 0;
                    
                    if (ex.Response != null)
                    {
                        using (var errorResponse = (HttpWebResponse)ex.Response)
                        using (var streamReader = new System.IO.StreamReader(errorResponse.GetResponseStream()))
                        {
                            string errorText = streamReader.ReadToEnd();
                            statusCode = errorResponse.StatusCode;
                            
                            if (errorResponse.StatusCode == HttpStatusCode.Unauthorized || 
                                errorResponse.StatusCode == HttpStatusCode.Forbidden)
                            {
                                errorDetail = errorText;
                                Log.Error($"[AI Summarizer] ? Authentication Error ({errorResponse.StatusCode}):");
                                Log.Error($"[AI Summarizer]    API Key: {SanitizeApiKey(apiKey)}");
                                Log.Error($"[AI Summarizer]    Provider: {provider}");
                                Log.Error($"[AI Summarizer]    Response: {errorText}");
                                Log.Error("[AI Summarizer] ");
                                Log.Error("[AI Summarizer] ? Possible solutions:");
                                Log.Error("[AI Summarizer]    1. Check if API Key is correct");
                                Log.Error("[AI Summarizer]    2. Verify Provider selection matches your key");
                                Log.Error("[AI Summarizer]    3. Check if API Key has sufficient credits");
                                Log.Error("[AI Summarizer]    4. Try regenerating your API Key");
                            }
                            else
                            {
                                errorDetail = errorText.Substring(0, Math.Min(200, errorText.Length));
                            }
                            
                            if (errorResponse.StatusCode == HttpStatusCode.ServiceUnavailable || 
                                errorResponse.StatusCode == (HttpStatusCode)429 ||              
                                errorResponse.StatusCode == HttpStatusCode.GatewayTimeout ||    
                                errorText.Contains("overloaded") ||
                                errorText.Contains("UNAVAILABLE"))
                            {
                                shouldRetry = true;
                            }
                            
                            if (errorResponse.StatusCode != HttpStatusCode.Unauthorized && 
                                errorResponse.StatusCode != HttpStatusCode.Forbidden)
                            {
                                Log.Warning($"[AI Summarizer] ? API Error (attempt {attempt}/{MAX_RETRIES}): {errorResponse.StatusCode} - {errorDetail}");
                            }
                        }
                    }
                    else
                    {
                        errorDetail = ex.Message;
                        Log.Warning($"[AI Summarizer] ? Network Error (attempt {attempt}/{MAX_RETRIES}): {errorDetail}");
                        shouldRetry = true;
                    }
                    
                    if (attempt >= MAX_RETRIES || !shouldRetry)
                    {
                        if (statusCode != HttpStatusCode.Unauthorized && 
                            statusCode != HttpStatusCode.Forbidden)
                        {
                            Log.Error($"[AI Summarizer] ? Failed after {attempt} attempts. Last error: {errorDetail}");
                        }
                        return null;
                    }
                    
                    await Task.Delay(RETRY_DELAY_MS * attempt);
                }
                catch (Exception ex)
                {
                    Log.Error($"[AI Summarizer] ? Unexpected error: {ex.GetType().Name} - {ex.Message}");
                    Log.Error($"[AI Summarizer]    Stack trace: {ex.StackTrace}");
                    return null;
                }
            }
            
            return null;
        }

        /// <summary>
        /// ? v3.3.21: ROBUST JSON PARSING using manual string parsing
        /// Replaces fragile Regex-based parsing with proper character-by-character parsing
        /// Handles escaped quotes (\"), newlines (\n), and other escape sequences
        /// </summary>
        private static string ParseJsonResponse(string responseText)
        {
            try
            {
                if (string.IsNullOrEmpty(responseText))
                {
                    Log.Error("[AI Summarizer] ? Empty response received");
                    return null;
                }
                
                if (provider == "Google")
                {
                    // Parse Google Gemini: {"candidates":[{"content":{"parts":[{"text":"..."}]}}]}
                    string result = ExtractJsonField(responseText, "text");
                    if (result != null)
                    {
                        return result;
                    }
                    
                    Log.Error("[AI Summarizer] ? Google response has no valid 'text' field");
                    if (Prefs.DevMode)
                    {
                        Log.Error($"[AI Summarizer] Response: {responseText.Substring(0, Math.Min(500, responseText.Length))}");
                    }
                }
                else
                {
                    // Parse OpenAI/DeepSeek/Player2/Custom: {"choices":[{"message":{"content":"..."}}]}
                    string result = ExtractJsonField(responseText, "content");
                    if (result != null)
                    {
                        return result;
                    }
                    
                    Log.Error("[AI Summarizer] ? OpenAI-format response has no valid 'content' field");
                    if (Prefs.DevMode)
                    {
                        Log.Error($"[AI Summarizer] Response: {responseText.Substring(0, Math.Min(500, responseText.Length))}");
                    }
                }
            }
            catch (Exception ex)
            {
                Log.Error($"[AI Summarizer] ? JSON parsing error: {ex.Message}");
                if (Prefs.DevMode)
                {
                    Log.Error($"[AI Summarizer] Stack: {ex.StackTrace}");
                }
            }
            
            return null;
        }
        
        /// <summary>
        /// ? v3.3.21: Extract JSON field value with proper escape handling
        /// Finds "fieldName": "value" and unescapes the value
        /// </summary>
        private static string ExtractJsonField(string json, string fieldName)
        {
            if (string.IsNullOrEmpty(json) || string.IsNullOrEmpty(fieldName))
                return null;
            
            // Search for "fieldName":
            string searchPattern = $"\"{fieldName}\"";
            int fieldIndex = json.IndexOf(searchPattern);
            if (fieldIndex < 0)
                return null;
            
            // Find the colon after field name
            int colonIndex = json.IndexOf(':', fieldIndex + searchPattern.Length);
            if (colonIndex < 0)
                return null;
            
            // Skip whitespace after colon
            int valueStart = colonIndex + 1;
            while (valueStart < json.Length && char.IsWhiteSpace(json[valueStart]))
            {
                valueStart++;
            }
            
            // Check if value is a string (starts with ")
            if (valueStart >= json.Length || json[valueStart] != '\"')
                return null;
            
            // Extract string value (handling escaped characters)
            return ExtractJsonString(json, valueStart + 1);
        }
        
        /// <summary>
        /// ? v3.3.21: Extract and unescape JSON string value
        /// Handles: \", \\, \n, \r, \t, \b, \f, \uXXXX
        /// </summary>
        private static string ExtractJsonString(string json, int startIndex)
        {
            var result = new StringBuilder();
            int i = startIndex;
            
            while (i < json.Length)
            {
                char c = json[i];
                
                if (c == '\"')
                {
                    // End of string
                    return result.ToString();
                }
                else if (c == '\\' && i + 1 < json.Length)
                {
                    // Escape sequence
                    i++;
                    char escaped = json[i];
                    
                    switch (escaped)
                    {
                        case '\"':
                            result.Append('\"');
                            break;
                        case '\\':
                            result.Append('\\');
                            break;
                        case '/':
                            result.Append('/');
                            break;
                        case 'n':
                            result.Append('\n');
                            break;
                        case 'r':
                            result.Append('\r');
                            break;
                        case 't':
                            result.Append('\t');
                            break;
                        case 'b':
                            result.Append('\b');
                            break;
                        case 'f':
                            result.Append('\f');
                            break;
                        case 'u':
                            // Unicode escape: \uXXXX
                            if (i + 4 < json.Length)
                            {
                                string hex = json.Substring(i + 1, 4);
                                try
                                {
                                    int codePoint = Convert.ToInt32(hex, 16);
                                    result.Append((char)codePoint);
                                    i += 4; // Skip the 4 hex digits
                                }
                                catch
                                {
                                    // Invalid unicode, keep as-is
                                    result.Append("\\u");
                                }
                            }
                            else
                            {
                                result.Append("\\u");
                            }
                            break;
                        default:
                            // Unknown escape, keep as-is
                            result.Append('\\');
                            result.Append(escaped);
                            break;
                    }
                    i++;
                }
                else
                {
                    // Regular character
                    result.Append(c);
                    i++;
                }
            }
            
            // String not properly closed
            return null;
        }
        
        // Player2 local app support
        private const string Player2LocalUrl = "http://localhost:4315/v1";
        private const string Player2RemoteUrl = "https://api.player2.game/v1";
        private const string Player2GameClientId = "rimtalk-expand-memory";
        private static bool isPlayer2Local = false;
        private static string player2LocalKey = null;
        
        public static void TryDetectPlayer2LocalApp()
        {
            Task.Run(async () =>
            {
                try
                {
                    Log.Message("[AI] ? Checking for local Player2 app...");
                    
                    var healthRequest = (HttpWebRequest)WebRequest.Create($"{Player2LocalUrl}/health");
                    healthRequest.Method = "GET";
                    healthRequest.Timeout = 2000;
                    
                    try
                    {
                        using (var response = (HttpWebResponse)await healthRequest.GetResponseAsync())
                        {
                            if (response.StatusCode == HttpStatusCode.OK)
                            {
                                Log.Message("[AI] ? Player2 local app detected!");
                                
                                await TryGetPlayer2LocalKey();
                                
                                if (!string.IsNullOrEmpty(player2LocalKey))
                                {
                                    isPlayer2Local = true;
                                    LongEventHandler.ExecuteWhenFinished(() =>
                                    {
                                        Messages.Message("RimTalk_Settings_Player2Detected".Translate(), MessageTypeDefOf.PositiveEvent, false);
                                    });
                                    return;
                                }
                            }
                        }
                    }
                    catch (WebException)
                    {
                        // Local app not running
                    }
                    
                    isPlayer2Local = false;
                    player2LocalKey = null;
                    Log.Message("[AI] ? Player2 local app not found, will use remote API");
                    LongEventHandler.ExecuteWhenFinished(() =>
                    {
                        Messages.Message("RimTalk_Settings_Player2NotFound".Translate(), MessageTypeDefOf.NeutralEvent, false);
                    });
                }
                catch (Exception ex)
                {
                    Log.Warning($"[AI] Player2 detection error: {ex.Message}");
                    isPlayer2Local = false;
                    player2LocalKey = null;
                }
            });
        }
        
        private static async Task TryGetPlayer2LocalKey()
        {
            try
            {
                string loginUrl = $"{Player2LocalUrl}/login/web/{Player2GameClientId}";
                
                var request = (HttpWebRequest)WebRequest.Create(loginUrl);
                request.Method = "POST";
                request.ContentType = "application/json";
                request.Timeout = 3000;
                
                byte[] bodyRaw = Encoding.UTF8.GetBytes("{}");
                request.ContentLength = bodyRaw.Length;
                
                using (var stream = await request.GetRequestStreamAsync())
                {
                    await stream.WriteAsync(bodyRaw, 0, bodyRaw.Length);
                }
                
                using (var response = (HttpWebResponse)await request.GetResponseAsync())
                using (var reader = new System.IO.StreamReader(response.GetResponseStream()))
                {
                    string responseText = await reader.ReadToEndAsync();
                    
                    // Simple string search for p2Key (more robust than regex for this simple case)
                    int keyIndex = responseText.IndexOf("\"p2Key\"");
                    if (keyIndex >= 0)
                    {
                        int colonIndex = responseText.IndexOf(":", keyIndex);
                        if (colonIndex >= 0)
                        {
                            int quoteStart = responseText.IndexOf("\"", colonIndex);
                            if (quoteStart >= 0)
                            {
                                int quoteEnd = responseText.IndexOf("\"", quoteStart + 1);
                                if (quoteEnd > quoteStart)
                                {
                                    player2LocalKey = responseText.Substring(quoteStart + 1, quoteEnd - quoteStart - 1);
                                    Log.Message($"[AI] ? Got Player2 local key: {SanitizeApiKey(player2LocalKey)}");
                                }
                            }
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                Log.Warning($"[AI] Failed to get Player2 local key: {ex.Message}");
                player2LocalKey = null;
            }
        }
    }
}
