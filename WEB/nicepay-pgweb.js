﻿﻿﻿﻿﻿var NicePayCommon = {};
var NicePayStd = {};

var merchantForm;
var existSubmitUrl;
var existTarget;
var hasFrame = false;
var documentBody;

var payMethod = "";
var nAgt= navigator.userAgent;
var disableScrollYN = "N";
var jsVer = "nicepay-pgweb";
var jsDeployedVer = "1.0.0";
var jsDeployedDate = "241031";

var nicepayDomain = "https://pg-web.nicepay.co.kr";
var ReqSubPath		= "/v3/gwPayment.jsp";

// direct call option
var POPUP = "POPUP";
/************************************
 * nicepay-gw.js를                  	*
 * 다운로드 또는 무단 수정하여              	*
 * 
 * 연동 시                            	*
 * 안정적인 결제 서비스를 보장할수 없습니다.    	* 
 ************************************/
function goPay(payForm){
	
	merchantForm = payForm;
	existSubmitUrl = payForm.action;
	
	// scroll prevention
	if(payForm.NPDisableScroll) {
		disableScrollYN = payForm.NPDisableScroll.value;
	}

	if(payForm.ConnWithIframe && payForm.ConnWithIframe.value=="Y"){
		if(top!=self && self.name!="") {
			hasFrame = true;
		}
	}
	
	if(hasFrame){
		documentBody = parent.document.body || parent.document.documentElement;
	}else{
		documentBody = document.body || document.documentElement;
	}
	
	try{
		var browser = NicePayStd.uaMatch(nAgt);
		
		if(payForm.MobileYN && payForm.MobileYN.value == "Y") {
			browser.mobile = true;
		}
		
		if(browser.mobile != true) {
			if(!document.body){
				alert("정상적이지 않은 형태로 결제창이 요청되었습니다. 가맹점으로 문의 해주시기 바랍니다.");
				NicePayCommon.errorReport("MJ04", "[MerchantJS]Merchant access fail document body");
				return false;
			}
			
			if(payForm.encoding.indexOf("multipart") > -1){
				alert("지원하지 않는 방식으로 결제창이 요청되었습니다. 가맹점으로 문의 해주시기 바랍니다.");
			    NicePayCommon.errorReport("MJ01", "[MerchantJS]Form encoding requested in multipart");
				return false;
			}
			
			
			if(browser.msie && browser.version < 9){
				alert("사용하시는 브라우저는 MicroSoft사의 보안지원이 종료된 버전입니다. \n 안전한 결제를 위해 Internet Explorer 9.0 이상 또는 크롬 등의 다른 브라우저 사용 요청 드립니다.");
			    NicePayCommon.errorReport("MJ02", "[MerchantJS]Payment requested on IE9 or lower");
				return false;
			}
			
			if(browser.msedge){
				// edge confirm check
				if(payForm.DisableEdgeChk && payForm.DisableEdgeChk.value == "Y"){
					// do nothing ~
				}else{
					if(browser.version < 17){
						var str_comfirm = "Win10 Edge 환경 결제 안내 \n\n Win10 Edge 환경에서는 결제가 원할 하지 않을 수 있습니다. \n\n"
							+ "결제를 이용하시려면 about:flags 창을 띄운 후 \n"
							+ "개발자 설정 > Microsoft 호환성 목록 사용 항목을 체크 해제로 \n"
							+ "설정하시면 결제를 진행하실 수 있습니다. \n\n"
							+ "해제를 설정하러 가시려면 '확인'을 \n"
							+ "결제를 진행하시려면 '취소'를 눌러주시기 바랍니다. \n\n";
						
						if(confirm(str_comfirm)){
							window.open("about:flags","optionWindow","width=500 , height=750");
							return;
						}
					}
				}
			}
			
			NicePayStd.setListener();
		}
		
		if(payForm.TransType) {
			if(payForm.TransType.value == "1"){
				var tempPayMethod = payForm.PayMethod.value.split(",");
				var isEscrow = true;
				var possible = "CARD,BANK,VBANK,";
					
				for(var i=0; i<tempPayMethod.length; i++){
					if(tempPayMethod[i] == ""){
						isEscrow = false;
						break;
					}
					if(possible.indexOf(tempPayMethod[i]+",") < 0){
						isEscrow = false;
						break;
					}
				}
				if(!isEscrow){
					alert("["+jsVer+"] 에스크로 서비스는 신용카드,계좌이체,가상계좌이체만 가능합니다.");
					return false;
				}
			}
		}
		
		//필수값 체크
		if(!payForm.MID || payForm.MID.value == "" || payForm.MID.value.length != 10) {
			alert("["+jsVer+"] MID 필드 길이가 올바르지 않습니다.");
			return false;
		}
		
		if(!payForm.Amt || payForm.Amt.value == "" || payForm.Amt.value.indexOf(".") > -1 || Number(payForm.Amt.value) == "NaN") {
			alert("["+jsVer+"] Amt 필드는 정수 값만 설정 가능합니다.");
			return false;
		}
		
		if(!payForm.EdiDate || payForm.EdiDate.value == "") {
			alert("["+jsVer+"] EdiDate 필드가 누락되었습니다.");
			return false;
		}
		
		if(!payForm.Moid || payForm.Moid.value == "") {
			alert("["+jsVer+"] Moid 필드가 누락되었습니다.");
			return false;
		}
		
		if(!payForm.SignData || payForm.SignData.value == "") {
			alert("["+jsVer+"] SignData 필드가 누락되었습니다.");
			return false;
		}
		
		//모바일 결제창 호출
		if(browser.mobile == true) {
			if(!payForm.ReturnURL || payForm.ReturnURL.value == "") {
				alert("["+jsVer+"] ReturnURL 필드가 누락되었습니다.");
				return false;
			}
		
			NicePayCommon.setFormData(payForm, "NpSvcType","SMART");
			payForm.target = "_top";
			payForm.method = "post";
			payForm.action = nicepayDomain+ReqSubPath;
			payForm.submit();
			return false;
		}
		
		var niceForm = payForm;
		NicePayCommon.setFormData(niceForm, "VerifySType","S");
		
		var NpDirectCard = [
			["02","459","456",""],
			["03","590","430",""],
			["04","400","400",""],
			["06","400","400",""],
			["07","400","400",""],
			["08","644","489",""],
			["11","644","480",""],
			["12","400","400",""],
			["15","644","510",""],
			["16","590","430",""],
			["17","644","510",""],
			["37","500","456",""],
			["44","644","480",""],
			["SSG","640","685",""],
			["SSG_MONEY","640","685",""],
			["SSGPAY_MONEY","640","685",""],
			["SamsungPay","440","630",""],
			["SAMSUNGPAYV2","400","750",""],
			["Payco","720","645",""],
			["KakaoPay","426","550",""],
			["SKPay","0","0","POPUP"],
			["SSGV2","640","685",""],
			["LPAY","450","470",""],
			["NaverPay","0","0","POPUP"],
			["APPLEPAY","0","0","POPUP"],
			["PINPAY","540","700",""],
			["TOSSPAY","0","0","POPUP"]
		];
		
		var NpDirectBank = [
			["LIIV","700","665",""],
			["KBANK","0","0","POPUP"],
		    ["BANK","0","0","POPUP"]
		];
		
		var NpDirectCellPhone = [
		    ["CELLPHONE","0","0","POPUP"],
		    ["SAMSUNGPAYV2","400","750",""]
		];
		
		var NpDirectSsgBank = [
 		    ["SSG_BANK","0","0","POPUP"]
 		];
		
		var NpDirectCmsBank = [
		    ["CMS_BANK","450","670",""]
		];
		
		var DirectShowOpt	= payForm.DirectShowOpt;
		var TransType		= payForm.TransType;
		var SelectCardCode	= payForm.SelectCardCode;
		var PayMethod		= payForm.PayMethod;
		var NicepayReserved = payForm.NicepayReserved;
		var DirectEasyPay 	= payForm.DirectEasyPay;
		var NpDirectEasyBank 	= payForm.DirectEasyBank;
		var DirectEasyHpp 	= payForm.DirectEasyHpp;
		var PayMethodValue	= "";
		
		var WindowWidth		= "660";
		var WindowHeight	= "505";
		var CallType		= "";
		
		var EscrowFlag		= false;
		var DirectFlag		= false;
		var DriectCardCd    = "";
		var NpDriectBankCd 	= "";
		var DirectCellPhoneCd	= "";
		
		// 에스크로 체크
		if(TransType) {
			if(TransType.value == "1") {
				EscrowFlag = true;
			}
		}

		// PayMethod 빈 값 체크
		if(PayMethod) {
			PayMethodValue = PayMethod.value;
			PayMethodArray = PayMethodValue.split(",");
			// 결제수단이 여러개일 경우 확인
			if(PayMethodArray.length > 1 || PayMethodValue == "") {
				PayMethodValue = "";
			}
		} else {
			PayMethodValue = "";
		}
		
		// DirectShowOpt 검증
		if(DirectShowOpt && !EscrowFlag && PayMethodValue != "") {
			var DirectShowArray = DirectShowOpt.value.split("|");
			// 최대 9개까지 처리하도록
			if(DirectShowArray.length > 0 && DirectShowArray.length < 10) {
				for(var i = 0; i < DirectShowArray.length; i++) {
					if(DirectShowArray[i] == PayMethodValue) {//CARD, BANK, CELLPHONE, SSG_BANK, CMS_BANK
						if(PayMethodValue == "CARD" && SelectCardCode){
							DriectCardCd = SelectCardCode.value;
						}
						if(PayMethodValue == "BANK"){
							NpDriectBankCd = "BANK";
						}
						if(PayMethodValue == "CELLPHONE"){
							DirectCellPhoneCd = "CELLPHONE";
						}
						DirectFlag = true;
						break;
					}else{//GIFT_SSG
						if(DirectShowArray[i] == "GIFT_SSG" && PayMethodValue == "CARD" && SelectCardCode.value=="SSG_MONEY"){
							DirectFlag = true;
							DriectCardCd = SelectCardCode.value;
							break;
						}
					}
				}
			} 
		}//end of if

		//NicepayReserved : easypay Direct Show vertify
		if(NicepayReserved){
			var NiceReservedArr = NicepayReserved.value.split("|");
			if(NiceReservedArr.length >0){
				for(var i=0;i<NiceReservedArr.length; i++){
					var ReservedParamArr = NiceReservedArr[i].split("=");
					if(ReservedParamArr[0] == "DirectSamsungPay" && ReservedParamArr[1] == "Y"){
						DriectCardCd = "SamsungPay";
						break;
					}
					if(ReservedParamArr[0] == "DirectPayco" && ReservedParamArr[1] == "Y"){
						DriectCardCd = "Payco";
						break;
					}
					if(ReservedParamArr[0] == "DirectKakao" && ReservedParamArr[1] == "Y"){
						DriectCardCd = "KakaoPay";
						break;
					}
					if(ReservedParamArr[0] == "DirectPay11" && ReservedParamArr[1] == "Y"){
						DriectCardCd = "SKPay";
						break;
					}
				}
			}
		}

		//DirectEasyPay : easypay Direct Show verify
		if(DirectEasyPay){
			if(DirectEasyPay.value == "E007"){
				DriectCardCd = "SSGV2";
			}else if(DirectEasyPay.value == "E018"){
				DriectCardCd = "LPAY";
			}else if(DirectEasyPay.value == "E020"){
				DriectCardCd = "NaverPay";
			}else if(DirectEasyPay.value == "E021"){
				DriectCardCd = "SAMSUNGPAYV2";
			}else if(DirectEasyPay.value == "E022"){
				DriectCardCd = "APPLEPAY";
			}else if(DirectEasyPay.value == "E025"){
				DriectCardCd = "TOSSPAY";
			}
		}
		
		if(DirectFlag) {
			if(PayMethodValue == "CARD") {
				if(DriectCardCd != ""){
					WindowWidth		= "440";
					WindowHeight	= "456";
					CallType		= "";
					for(var i=0;i < NpDirectCard.length;i++){
						if(NpDirectCard[i][0] == DriectCardCd){
							WindowWidth		= NpDirectCard[i][1];
							WindowHeight	= NpDirectCard[i][2];
							CallType		= NpDirectCard[i][3];
							break;
						}
					}
				}else{
					DirectFlag = false;
					DirectShowOpt.value = "";
				}
			} else if(PayMethodValue == "BANK") {
				//NpDirectEasyBank : easybank Direct Show verify
				if(NpDirectEasyBank && NpDirectEasyBank.value != ""){
					if(NpDirectEasyBank.value == "E019"){
						NpDriectBankCd = "LIIV";
					}else if(NpDirectEasyBank.value == "E023"){
						NpDriectBankCd = "KBANK";
					}else{
						NpDriectBankCd = "";
					}
				}
				
				if(NpDriectBankCd != ""){
					for(var i=0;i < NpDirectBank.length;i++){
						if(NpDirectBank[i][0] == NpDriectBankCd){
							WindowWidth		= NpDirectBank[i][1];
							WindowHeight	= NpDirectBank[i][2];
							CallType		= NpDirectBank[i][3];
							break;
						}
					}
				}else{
					DirectFlag = false;
					DirectShowOpt.value = "";
				}
			} else if(PayMethodValue == "CELLPHONE") {
				if(DirectEasyHpp && DirectEasyHpp.value != "") {
					if(DirectEasyHpp.value == "E021") {
						DirectCellPhoneCd = "SAMSUNGPAYV2";
					} else {
						DirectCellPhoneCd = "";
					}
				}
				
				if(DirectCellPhoneCd != ""){
					for(var i=0;i < NpDirectCellPhone.length;i++){
						if(NpDirectCellPhone[i][0] == DirectCellPhoneCd){
							WindowWidth		= NpDirectCellPhone[i][1];
							WindowHeight	= NpDirectCellPhone[i][2];
							CallType		= NpDirectCellPhone[i][3];
							break;
						}
					}
				}else{
					DirectFlag = false;
					DirectShowOpt.value = "";
				}
			} else if(PayMethodValue == "SSG_BANK") {
				for(var i=0;i < NpDirectSsgBank.length;i++){
					if(NpDirectSsgBank[i][0] == "SSG_BANK"){
						WindowWidth		= NpDirectSsgBank[i][1];
						WindowHeight	= NpDirectSsgBank[i][2];
						CallType		= NpDirectSsgBank[i][3];
						break;
					}
				}
			} else if(PayMethodValue == "CMS_BANK") {
				for(var i=0;i < NpDirectCmsBank.length;i++){
					if(NpDirectCmsBank[i][0] == "CMS_BANK"){
						WindowWidth		= NpDirectCmsBank[i][1];
						WindowHeight	= NpDirectCmsBank[i][2];
						CallType		= NpDirectCmsBank[i][3];
						break;
					}
				}
			} else {// 카드, 계좌이체, 휴대폰을 제외한 결제수단은 일반 결제창 호출
				WindowWidth		= "660";
				WindowHeight	= "505";
				CallType		= "";
				DirectFlag = false;
			}
		} else {
			if(DirectShowOpt) {// 바로호출이 아닐 경우 초기화
				DirectShowOpt.value = "";
			}
		}

		NicePayStd.creatLayer(WindowWidth, WindowHeight, CallType);
		niceForm.action = nicepayDomain+ReqSubPath;
		NicePayCommon.disableScroll();
		//페이지 레이어 설정
		existTarget = payForm.target;
		niceForm.target = 'nice_frame';
		NicePayCommon.setFormData(niceForm, "EncGoodsName", escape(niceForm.GoodsName.value));
		NicePayCommon.setFormData(niceForm, "EncBuyerName", (niceForm.BuyerName)? escape(niceForm.BuyerName.value):"");
		NicePayCommon.setFormData(niceForm, "NpDirectYn", (DirectFlag==true)? "Y" : "N");
		NicePayCommon.setFormData(niceForm, "NpDirectLayer", "Y");
		NicePayCommon.setFormData(niceForm, "NpForwardNew", "Y");
		NicePayCommon.setFormData(niceForm, "JsVer", jsVer);
		NicePayCommon.setFormData(niceForm, "NpSvcType", "WEBSTD");
		NicePayCommon.setFormData(niceForm, "DeployedVer", jsDeployedVer);
		NicePayCommon.setFormData(niceForm, "DeployedDate", jsDeployedDate);
		NicePayCommon.setFormData(niceForm, "DeployedFileName", jsVer);
		
		niceForm.submit();	
	}catch(e){
		alert("결제 요청 중 에러가 발생하였습니다. 다시 시도해 주십시오.");
		NicePayCommon.errorReport("MJ03", "[MerchantJS]Exception : "+e.message);
		return false;
	}
}

function deleteLayer(){
	NicePayStd.deleteLayer();
}
/////////////////////////////////////////////////////////////////////
// NicePay Common Function
/////////////////////////////////////////////////////////////////////
NicePayCommon.createDivElement = function(frameID, zindex){
	if(document.getElementById(frameID)==null){
		var newDiv = (hasFrame==false) ? document.createElement('div') : parent.document.createElement('div');
		
		newDiv.id = frameID;
		newDiv.style.position = "absolute";
		newDiv.style.zIndex = zindex;
		newDiv.style.clear = "both";
		newDiv.style.top = 0;
		newDiv.style.left = 0;
		documentBody.appendChild(newDiv);
	}
}
NicePayCommon.disableScroll = function(){
	if(disableScrollYN == "Y"){
		if( document.body  && document.body.scrollTop) {
		 	document.body.style.overflowX = ""; 
		} else if( document.documentElement ) {
			document.documentElement.style.overflowX  = ""; 
		}
		documentBody.style.overflow = "hidden"; 
	}else{
		if( document.body  && document.body.scrollTop) {
		 	document.body.style.overflowX = "hidden"; 
		} else if( document.documentElement ) {
			document.documentElement.style.overflowX  = "hidden"; 
		}
	} 
}
NicePayCommon.enableScroll = function(){
	if(disableScrollYN == "Y"){
		if( document.body && document.body.style.overflow) {
			document.body.style.overflowX  = ""; 
		} else if( document.documentElement && document.documentElement.style.overflow ) {
			document.documentElement.style.overflowX  = ""; 
		}
		documentBody.style.overflow  = "auto";
	}else{
		if( document.body && document.body.style.overflow) {
			document.body.style.overflowX  = "auto"; 
		} else if( document.documentElement && document.documentElement.style.overflow ) {
			document.documentElement.style.overflowX  = "auto"; 
		}
	}
}
NicePayCommon.setFormData = function(f, key, value){
	if(f[key]){
		f[key].value = value;
	}else{
		var input = document.createElement("input");
		input.type = "hidden";
		input.name = key;
		input.value =  value;
		f.appendChild(input);
	}
}
NicePayCommon.errorReport = function(errorCd, errorMsg){
	try{
		var newForm = document.createElement('form');
		newForm.id = "error_report";
		newForm.name = "error_report";
		newForm.method = "post";
		newForm.target = "error_frame";
		newForm.action = nicepayDomain + "/v3/api/errorReport.jsp";
		documentBody.appendChild(newForm);
		
		var newIframe = document.createElement('iframe');
		newIframe.id = "error_frame";
		newIframe.name = "error_frame";
		newIframe.width = "0px";
		newIframe.height = "0px";
		documentBody.appendChild(newIframe);
		
		var errorMsg = {
			"MID" : merchantForm.MID.value,
			"URL" : window.location.href,
			"ERR_CD" : errorCd,
			"ERR_MSG" : escape(errorMsg),
			"PayMethod" : merchantForm.PayMethod.value,
			"Worker" : "WEBSTD",
			"SvcType" : "01",
			"Moid" : merchantForm.Moid.value,
			"JsVersion" : jsVer+"|"+jsDeployedVer+"|"+jsDeployedDate
		};

		for(var key in errorMsg){
			var newInput = document.createElement('input');
			newInput.name = key;
			newInput.id = key;
			newInput.value = errorMsg[key];
			newInput.type = "hidden";
			newForm.appendChild(newInput);
		}
		newForm.submit();
		
		setTimeout(function() {
			documentBody.removeChild(document.getElementById("error_report"));
			documentBody.removeChild(document.getElementById("error_frame"));
		}, 1000);
	}catch(e){
	}
}
/////////////////////////////////////////////////////////////////////
// WEB STD Function
/////////////////////////////////////////////////////////////////////
NicePayStd.receiveMessageValue = function(e) {
	merchantForm.action = existSubmitUrl;
	merchantForm.target = existTarget;
	if (window.postMessage) {
		//var obj = JSON.parse(e.data);
		var obj = e.data;
		if (obj.code) {
			// obj.success is defined. We received the response via postMessage
			// as an object, and we are using a browser that supports
			// postMessage objects (not IE8/9).
			NicePayStd.resultMessage(obj);
		} else {
			// We received the response via postMessage as a string. This works
			// for all browsers that support postMessage, including IE8/9.
			var str = eval("(" + obj + ")");
			NicePayStd.resultMessage(str);
		}
	} else {
		//alert('postMessage not supported');
		NicePayStd.resultMessage({"code" : "1", "message" : "postMessage not supported"});
	}
}
NicePayStd.resultMessage = function(obj){
	switch(obj.code){
		case "0" :
			//success
			if(obj.result){
				NicePayCommon.setFormData(merchantForm, "AuthResultCode", "0000");
				NicePayCommon.setFormData(merchantForm, "AuthResultMsg", "인증 성공");
				NicePayCommon.setFormData(merchantForm, "AuthToken", obj.result.AuthToken);
				NicePayCommon.setFormData(merchantForm, "TxTid", obj.result.TxTid);
				if(obj.result.BillAuthYN == "Y") {
					NicePayCommon.setFormData(merchantForm, "BillAuthYN", "Y");
					if(obj.result.BillingNextAppURL != "") {
						NicePayCommon.setFormData(merchantForm, "NextAppURL", obj.result.BillingNextAppURL);
					}
				} else {
					NicePayCommon.setFormData(merchantForm, "NextAppURL", obj.result.NextAppURL);
					NicePayCommon.setFormData(merchantForm, "NetCancelURL", obj.result.NetCancelURL);
				}
				
				if(obj.result.ServiceIDC != "") {
					NicePayCommon.setFormData(merchantForm, "AuthFlg", obj.result.AuthFlg);
					NicePayCommon.setFormData(merchantForm, "BuyerAuthNum", obj.result.BuyerAuthNum);
					NicePayCommon.setFormData(merchantForm, "ServiceIDC", obj.result.ServiceIDC);
				}
				
				NicePayCommon.setFormData(merchantForm, "Signature", obj.result.Signature);
				
				NicePayCommon.setFormData(merchantForm, "EncGoodsName", "");
				NicePayCommon.setFormData(merchantForm, "EncBuyerName", "");

				nicepaySubmit();
				if(merchantForm.RmAuthLayer && merchantForm.RmAuthLayer.value=="Y"){
					NicePayStd.deleteLayer();
				}else{
					NicePayStd.deletePayment();
				}
			}else{
				resultMessage({"code" : "1", "message" : "not result value"});
			}
			break;
		case "1" :
			if(obj.result){
				NicePayCommon.setFormData(merchantForm, "AuthResultCode", obj.result.AuthResultCode);
				NicePayCommon.setFormData(merchantForm, "AuthResultMsg", obj.result.AuthResultMsg);
				NicePayCommon.setFormData(merchantForm, "AuthToken", obj.result.AuthToken);
				NicePayCommon.setFormData(merchantForm, "Signature", obj.result.Signature);
			}else{
				NicePayCommon.setFormData(merchantForm, "AuthResultCode", "9999");
				NicePayCommon.setFormData(merchantForm, "AuthResultMsg", obj.message);
			}
			//iframe 삭제
			NicePayStd.deleteLayer();
			//cancel
			try{
				nicepayClose();
			}catch(e){
			}
			break;
		case "2" :
			if(obj.result){
				NicePayCommon.setFormData(merchantForm, "AuthResultCode", obj.result.AuthResultCode);
				NicePayCommon.setFormData(merchantForm, "AuthResultMsg", obj.result.AuthResultMsg);
				NicePayCommon.setFormData(merchantForm, "AuthToken", obj.result.AuthToken);
				NicePayCommon.setFormData(merchantForm, "Signature", obj.result.Signature);
			}else{
				NicePayCommon.setFormData(merchantForm, "AuthResultCode", "9999");
				NicePayCommon.setFormData(merchantForm, "AuthResultMsg", obj.message);
			}
			//iframe 삭제
			NicePayStd.deleteLayer();
			try{
				nicepayClose();
			}catch(e){
			}
			break;
		case "10" :
			var payLayer = (hasFrame == false) ? document.getElementById("nice_layer") : parent.document.getElementById("nice_layer");
			payLayer.style.width = obj.width + "px";
			break;
		case "11" :
			var payLayer = (hasFrame == false) ? document.getElementById("nice_layer") : parent.document.getElementById("nice_layer");
			var niceFrame = (hasFrame == false) ? document.getElementById("nice_frame") : parent.document.getElementById("nice_frame");
			payLayer.style.width = obj.width + "px";
			payLayer.style.height = obj.height + "px";
			niceFrame.style.width = obj.width + "px";
			niceFrame.style.height = obj.height + "px";
			niceFrame.style.borderRadius = "15px";
			NicePayStd.niceLayerHandler();
			break;
		default :
			//코드별 에러 처리
			//obj.code
			//obj.message
			break;
	}
}
//레이어생성
NicePayStd.creatLayer = function(w, h, opt){
	//************** 레이어 동적생성 *********************//
	NicePayCommon.createDivElement("nice_layer", 9999);
	NicePayCommon.createDivElement("bg_layer", 9998);
	//iframe 동적 생성
	var html = "";
	html += "<div style=\"width:100%;text-align:center;padding:95px 0 0 0;\">";
	
	if(opt != POPUP) {
		html += "	<div><span style=\"color:#FFF;font-size:15px;\">Please, wait...</span></div>";
		html += "	<div style=\"position: relative;top: -180px;\"><img src=\""+nicepayDomain+"/webstd/images/loading.gif\" style=\"width: 344px;\"></div>";
	} else {
		html += "	<div style=\"position: relative;top: -400px;left:-200px;\"><img src=\""+nicepayDomain+"/webstd/images/loading.gif\" style=\"width: 344px;\"></div>";
	}
	html += "</div>";
	
	html += "<div id=\"payment_layer\" style=\"position: absolute; width: 100%; top: 0;\">";
	html += "	<iframe name=\"nice_frame\" id=\"nice_frame\" src=\"\" width=\"100%\" height=\""+h+"px\" scrolling=\"no\" frameborder=\"no\"></iframe>";
	html += "</div>";
	
	var payLayer = (hasFrame == false) ? document.getElementById("nice_layer") : parent.document.getElementById("nice_layer");
	var bgLayer = (hasFrame == false) ? document.getElementById("bg_layer") : parent.document.getElementById("bg_layer");
	
	payLayer.innerHTML = html;
	if(payLayer!=null){
		payLayer.style.top = (NicePayStd.getWindowHeight() - h) / 2 + NicePayStd.getYposition() + "px";
		payLayer.style.left = (NicePayStd.getWindowWidth() - w) / 2 + NicePayStd.getXposition() + "px";
	}
	
	payLayer.style.width = w + "px";
	payLayer.style.height = h + "px";
	bgLayer.style.width = "100%";
	bgLayer.style.height = documentBody.scrollHeight +"px";
	bgLayer.style.background = "#4c4c4c";

	var opacity = 65;
	if(merchantForm.Opacity && merchantForm.Opacity.value != ""){
		opacity = merchantForm.Opacity.value;
	}
	bgLayer.style.filter="alpha(opacity="+opacity+")";
	bgLayer.style.opacity = opacity/100;
	try{
		if(merchantForm.OptionList!=null && merchantForm.OptionList!="undefined"){
		var optionVal = merchantForm.OptionList.value;
			if(optionVal.indexOf("hidden")!=-1){
				nice_layer.style.filter="alpha(opacity=0)";
				bgLayer.style.filter="alpha(opacity=0)";
			}
		}
	}catch(e){
	}
}

//레이어 동적생성삭제
NicePayStd.deleteLayer = function(){
	NicePayCommon.enableScroll();
	
	if(hasFrame){
		documentBody.removeChild(parent.document.getElementById("nice_layer"));
		documentBody.removeChild(parent.document.getElementById("bg_layer"));
	}else{
		documentBody.removeChild(document.getElementById("nice_layer"));
		documentBody.removeChild(document.getElementById("bg_layer"));
	}
	NicePayStd.removeListener();
}

NicePayStd.deletePayment = function(){
	var nicelayer = (hasFrame==false) ? document.getElementById("nice_layer") : parent.document.getElementById("nice_layer");
	if(hasFrame){
		nicelayer.removeChild(parent.document.getElementById("payment_layer"));
	}else{
		nicelayer.removeChild(document.getElementById("payment_layer"));
	}
	NicePayStd.removeListener();
}

NicePayStd.getYposition = function(){
	  var scrollY = 0;
	  if(hasFrame){
		  if( typeof( parent.window.pageYOffset ) == 'number' ) {
			  scrollY = parent.window.pageYOffset;
		  } else if( parent.document.body && ( parent.document.body.scrollTop ) ) {
			  scrollY =  parent.document.body.scrollTop;
		  } else if( parent.document.documentElement && ( parent.document.documentElement.scrollTop ) ) {
			  scrollY =  parent.document.documentElement.scrollTop;
		  }
	  }else{
		  if( typeof( window.pageYOffset ) == 'number' ) {
			  scrollY = window.pageYOffset;
		  } else if( document.body && ( document.body.scrollTop ) ) {
			  scrollY = (hasFrame==false) ?  document.body.scrollTop : parent.document.body.scrollTop;
		  } else if( document.documentElement && ( document.documentElement.scrollTop ) ) {
			  scrollY = (hasFrame==false) ? document.documentElement.scrollTop : parent.document.documentElement.scrollTop;
		  }
	  }
  
	  return scrollY ;
}

NicePayStd.getXposition = function(){
	var scrollX = 0;
	if(hasFrame){
		if( typeof( parent.window.pageXOffset ) == 'number' ) {
			scrollX = parent.window.pageXOffset;
		} else if( parent.document.body && ( parent.document.body.scrollLeft ) ) {
			scrollX =  parent.document.body.scrollLeft;
		} else if( parent.document.documentElement && ( parent.document.documentElement.scrollLeft ) ) {
			scrollX =  parent.document.documentElement.scrollLeft;
		}
	}else{
		if( typeof( window.pageXOffset ) == 'number' ) {
			scrollX = window.pageXOffset;
		} else if( document.body && ( document.body.scrollLeft ) ) {
			scrollX = (hasFrame==false) ?  document.body.scrollLeft : parent.document.body.scrollLeft;
		} else if( document.documentElement && ( document.documentElement.scrollLeft ) ) {
			scrollX = (hasFrame==false) ? document.documentElement.scrollLeft : parent.document.documentElement.scrollLeft;
		}
	}
	return scrollX ;
}

NicePayStd.getWindowHeight = function() {
	  var myHeight = 0;
	  if(hasFrame){
		  if( typeof( parent.window.innerWidth ) == 'number' ) {
			  //Non-IE
			  myHeight = parent.window.innerHeight;
		  } else if( parent.document.documentElement && parent.document.documentElement.clientHeight ) {
			  //IE 6+ in 'standards compliant mode'
			  myHeight = parent.document.documentElement.clientHeight;
		  } else if( parent.document.body && parent.document.body.clientHeight ) {
			  //IE 4 compatible
			  myHeight = parent.document.body.clientHeight;
		  }
	  }else{
		  if( typeof( window.innerWidth ) == 'number' ) {
			  //Non-IE
			  myHeight = window.innerHeight;
		  } else if( document.documentElement && document.documentElement.clientHeight ) {
			  //IE 6+ in 'standards compliant mode'
			  myHeight = document.documentElement.clientHeight;
		  } else if( document.body && document.body.clientHeight ) {
			  //IE 4 compatible
			  myHeight = document.body.clientHeight;
		  }
	  }
	  return myHeight;
}

NicePayStd.getWindowWidth = function() {
	var myWidth = 0;
	if(hasFrame){
		if( typeof( parent.window.innerWidth ) == 'number' ) {
			//Non-IE
			myWidth = parent.window.innerWidth;
		} else if( parent.document.documentElement && parent.document.documentElement.clientWidth ) {
			//IE 6+ in 'standards compliant mode'
			myWidth = parent.document.documentElement.clientWidth;
		} else if( parent.document.body && parent.document.body.clientWidth ) {
			//IE 4 compatible
			myWidth = parent.document.body.clientWidth;
		}
	}else{
		if( typeof( window.innerWidth ) == 'number' ) {
			//Non-IE
			myWidth = window.innerWidth;
		} else if( document.documentElement && document.documentElement.clientWidth ) {
			//IE 6+ in 'standards compliant mode'
			myWidth = document.documentElement.clientWidth;
		} else if( document.body && document.body.clientWidth ) {
			//IE 4 compatible
			myWidth = document.body.clientWidth;
		}
	}
	return myWidth;
}

NicePayStd.niceLayerHandler = function(){
	try{
		var nicelayer = (hasFrame==false) ? document.getElementById("nice_layer") : parent.document.getElementById("nice_layer");
		if(nicelayer!=null){
			nicelayer.style.top = (NicePayStd.getWindowHeight() - nicelayer.style.height.replace("px",""))/2 + NicePayStd.getYposition() +"px";
			nicelayer.style.left = (NicePayStd.getWindowWidth() - nicelayer.style.width.replace("px",""))/2 + NicePayStd.getXposition() +"px";
		}
	}catch(e){
	}
}

NicePayStd.setListener = function(){
	if(hasFrame){
		if (parent.window.addEventListener) {  // all browsers except IE before version 9
			parent.window.addEventListener("message", NicePayStd.receiveMessageValue, false);
		} else {
			if (parent.window.attachEvent) {   // IE before version 9
				parent.window.attachEvent("onmessage", NicePayStd.receiveMessageValue);     // Internet Explorer from version 8
			}
		}
		if(typeof parent.window.addEventListener != "undefined")	parent.window.addEventListener("resize",NicePayStd.niceLayerHandler, false);
		if(typeof parent.window.attachEvent != "undefined" )	parent.window.attachEvent("onresize",NicePayStd.niceLayerHandler);
		
		if(typeof parent.window.addEventListener != "undefined")	parent.window.addEventListener("scroll",NicePayStd.niceLayerHandler, false);
		if(typeof parent.window.attachEvent != "undefined" )	parent.window.attachEvent("onscroll",NicePayStd.niceLayerHandler);
	}else{
		if (window.addEventListener) {  // all browsers except IE before version 9
			window.addEventListener("message", NicePayStd.receiveMessageValue, false);
		} else {
			if (window.attachEvent) {   // IE before version 9
				window.attachEvent("onmessage", NicePayStd.receiveMessageValue);     // Internet Explorer from version 8
			}
		}
		if(typeof window.addEventListener != "undefined")	window.addEventListener("resize",NicePayStd.niceLayerHandler, false);
		if(typeof window.attachEvent != "undefined" )	window.attachEvent("onresize",NicePayStd.niceLayerHandler);
		
		if(typeof window.addEventListener != "undefined")	window.addEventListener("scroll",NicePayStd.niceLayerHandler, false);
		if(typeof window.attachEvent != "undefined" )	window.attachEvent("onscroll",NicePayStd.niceLayerHandler);
	}
}

NicePayStd.removeListener = function(){
	try{
		if(hasFrame){
			if (parent.window.addEventListener) {  // all browsers except IE before version 9
				parent.window.removeEventListener("message", NicePayStd.receiveMessageValue, false);
			} else {
				if (parent.window.attachEvent) {   // IE before version 9
					parent.window.detachEvent("onmessage", NicePayStd.receiveMessageValue);     // Internet Explorer from version 8
				}
			}
			if(typeof parent.window.addEventListener != "undefined")	parent.window.removeEventListener("resize",NicePayStd.niceLayerHandler, false);
			if(typeof parent.window.attachEvent != "undefined" )	parent.window.detachEvent("onresize",NicePayStd.niceLayerHandler);
			
			if(typeof parent.window.addEventListener != "undefined")	parent.window.removeEventListener("scroll",NicePayStd.niceLayerHandler, false);
			if(typeof parent.window.attachEvent != "undefined" )	parent.window.detachEvent("onscroll",NicePayStd.niceLayerHandler);
		}else{
			if (window.addEventListener) {  // all browsers except IE before version 9
				window.removeEventListener("message", NicePayStd.receiveMessageValue, false);
			} else {
				if (window.attachEvent) {   // IE before version 9
					window.detachEvent("onmessage", NicePayStd.receiveMessageValue);     // Internet Explorer from version 8
				}
			}
			if(typeof window.addEventListener != "undefined")	window.removeEventListener("resize",NicePayStd.niceLayerHandler, false);
			if(typeof window.attachEvent != "undefined" )	window.detachEvent("onresize",NicePayStd.niceLayerHandler);
			
			if(typeof window.addEventListener != "undefined")	window.removeEventListener("scroll",NicePayStd.niceLayerHandler, false);
			if(typeof window.attachEvent != "undefined" )	window.detachEvent("onscroll",NicePayStd.niceLayerHandler);
		}
	}catch(e){}
}

NicePayStd.uaMatch = function(ua) {
	// If an UA is not provided, default to the current browser UA.
	if (ua === undefined) {
		ua = window.navigator.userAgent;
	}
	ua = ua.toLowerCase();

	var match = /(edge)\/([\w.]+)/.exec(ua)
			|| /(opr)[\/]([\w.]+)/.exec(ua)
			|| /(chrome)[ \/]([\w.]+)/.exec(ua)
			|| /(iemobile)[\/]([\w.]+)/.exec(ua)
			|| /(version)(applewebkit)[ \/]([\w.]+).*(safari)[ \/]([\w.]+)/.exec(ua)
			|| /(webkit)[ \/]([\w.]+).*(version)[ \/]([\w.]+).*(safari)[ \/]([\w.]+)/.exec(ua) 
			|| /(webkit)[ \/]([\w.]+)/.exec(ua)
			|| /(opera)(?:.*version|)[ \/]([\w.]+)/.exec(ua)
			|| /(msie) ([\w.]+)/.exec(ua)
			|| ua.indexOf("trident") >= 0 && /(rv)(?::| )([\w.]+)/.exec(ua)
			|| ua.indexOf("compatible") < 0 && /(mozilla)(?:.*? rv:([\w.]+)|)/.exec(ua)
			|| [];

	var platform_match = /(ipad)/.exec(ua) || /(ipod)/.exec(ua)
			|| /(windows phone)/.exec(ua) || /(iphone)/.exec(ua)
			|| /(kindle)/.exec(ua) || /(silk)/.exec(ua)
			|| /(android)/.exec(ua) || /(win)/.exec(ua)
			|| /(mac)/.exec(ua) || /(linux)/.exec(ua)
			|| /(cros)/.exec(ua) || /(playbook)/.exec(ua)
			|| /(bb)/.exec(ua) || /(blackberry)/.exec(ua) 
			|| [];

	var browser = {}, matched = {
		browser : match[5] || match[3] || match[1] || "",
		version : match[2] || match[4] || "0",
		versionNumber : match[4] || match[2] || "0",
		platform : platform_match[0] || ""
	};

	if (matched.browser) {
		browser[matched.browser] = true;
		browser.version = matched.version;
		browser.versionNumber = parseInt(matched.versionNumber, 10);
	}

	if (matched.platform) {
		browser[matched.platform] = true;
	}

	// These are all considered mobile platforms, meaning they run a
	// mobile browser
	if (browser.android || browser.bb || browser.blackberry
			|| browser.ipad || browser.iphone || browser.ipod
			|| browser.kindle || browser.playbook || browser.silk
			|| browser["windows phone"]) {
		browser.mobile = true;
	}

	// These are all considered desktop platforms, meaning they run
	// a desktop browser
	if (browser.cros || browser.mac || browser.linux || browser.win) {
		browser.desktop = true;
	}

	// Chrome, Opera 15+ and Safari are webkit based browsers
	if (browser.chrome || browser.opr || browser.safari) {
		browser.webkit = true;
	}

	// IE11 has a new token so we will assign it msie to avoid
	// breaking changes
	if (browser.rv || browser.iemobile) {
		var ie = "msie";

		matched.browser = ie;
		browser[ie] = true;
	}

	// Edge is officially known as Microsoft Edge, so rewrite the
	// key to match
	if (browser.edge) {
		delete browser.edge;
		var msedge = "msedge";

		matched.browser = msedge;
		browser[msedge] = true;
	}

	// Blackberry browsers are marked as Safari on BlackBerry
	if (browser.safari && browser.blackberry) {
		var blackberry = "blackberry";

		matched.browser = blackberry;
		browser[blackberry] = true;
	}

	// Playbook browsers are marked as Safari on Playbook
	if (browser.safari && browser.playbook) {
		var playbook = "playbook";

		matched.browser = playbook;
		browser[playbook] = true;
	}

	// BB10 is a newer OS version of BlackBerry
	if (browser.bb) {
		var bb = "blackberry";

		matched.browser = bb;
		browser[bb] = true;
	}

	// Opera 15+ are identified as opr
	if (browser.opr) {
		var opera = "opera";

		matched.browser = opera;
		browser[opera] = true;
	}

	// Stock Android browsers are marked as Safari on Android.
	if (browser.safari && browser.android) {
		var android = "android";

		matched.browser = android;
		browser[android] = true;
	}

	// Kindle browsers are marked as Safari on Kindle
	if (browser.safari && browser.kindle) {
		var kindle = "kindle";

		matched.browser = kindle;
		browser[kindle] = true;
	}

	// Kindle Silk browsers are marked as Safari on Kindle
	if (browser.safari && browser.silk) {
		var silk = "silk";

		matched.browser = silk;
		browser[silk] = true;
	}

	// Assign the name and platform variable
	browser.name = matched.browser;
	browser.platform = matched.platform;

	return browser;
}
