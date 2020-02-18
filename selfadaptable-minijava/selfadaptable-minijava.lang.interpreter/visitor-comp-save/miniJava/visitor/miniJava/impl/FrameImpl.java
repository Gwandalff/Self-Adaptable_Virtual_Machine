package miniJava.visitor.miniJava.impl;

import java.lang.IllegalArgumentException;
import java.lang.Object;
import java.lang.Override;
import miniJava.visitor.miniJava.Call;
import miniJava.visitor.miniJava.Context;
import miniJava.visitor.miniJava.Frame;
import miniJava.visitor.miniJava.MiniJavaPackage;
import miniJava.visitor.miniJava.ObjectInstance;
import miniJava.visitor.miniJava.Value;
import org.eclipse.emf.common.notify.Notification;
import org.eclipse.emf.common.notify.NotificationChain;
import org.eclipse.emf.ecore.EClass;
import org.eclipse.emf.ecore.InternalEObject;
import org.eclipse.emf.ecore.impl.ENotificationImpl;
import org.eclipse.emf.ecore.impl.MinimalEObjectImpl;
import org.eclipse.emf.ecore.util.EcoreUtil;

import visitor.IDynamicModule;
import visitor.VisitorInterface;

public class FrameImpl extends MinimalEObjectImpl.Container implements Frame {
	protected Call call;

	protected ObjectInstance instance;

	protected Frame childFrame;

	protected Context rootContext;

	protected Value returnValue;

	protected FrameImpl() {
		super();
	}

	@Override
	protected EClass eStaticClass() {
		return MiniJavaPackage.Literals.FRAME;
	}

	public Call getCall() {
		return call;
	}

	public NotificationChain basicSetCall(Call newCall, NotificationChain msgs) {
		Call oldCall = call;
		call = newCall;
		if (eNotificationRequired()) {
			ENotificationImpl notification = new ENotificationImpl(this, Notification.SET, MiniJavaPackage.FRAME__CALL, oldCall, newCall);
			if (msgs == null)
				msgs = notification;
			else
				msgs.add(notification);
		}
		return msgs;
	}

	public void setCall(Call newCall) {
		if (newCall != call) {
			NotificationChain msgs = null;
			if (call != null)
				msgs = ((InternalEObject) call).eInverseRemove(this, EOPPOSITE_FEATURE_BASE - MiniJavaPackage.FRAME__CALL, null, msgs);
			if (newCall != null)
				msgs = ((InternalEObject) newCall).eInverseAdd(this, EOPPOSITE_FEATURE_BASE - MiniJavaPackage.FRAME__CALL, null, msgs);
			msgs = basicSetCall(newCall, msgs);
			if (msgs != null)
				msgs.dispatch();
		} else if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, MiniJavaPackage.FRAME__CALL, newCall, newCall));
	}

	public ObjectInstance getInstance() {
		if (instance != null && instance.eIsProxy()) {
			InternalEObject oldInstance = (InternalEObject) instance;
			instance = (ObjectInstance) eResolveProxy(oldInstance);
			if (instance != oldInstance) {
				if (eNotificationRequired())
					eNotify(new ENotificationImpl(this, Notification.RESOLVE, MiniJavaPackage.FRAME__INSTANCE, oldInstance, instance));
			}
		}
		return instance;
	}

	public ObjectInstance basicGetInstance() {
		return instance;
	}

	public void setInstance(ObjectInstance newInstance) {
		ObjectInstance oldInstance = instance;
		instance = newInstance;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, MiniJavaPackage.FRAME__INSTANCE, oldInstance, instance));
	}

	public Frame getChildFrame() {
		return childFrame;
	}

	public NotificationChain basicSetChildFrame(Frame newChildFrame, NotificationChain msgs) {
		Frame oldChildFrame = childFrame;
		childFrame = newChildFrame;
		if (eNotificationRequired()) {
			ENotificationImpl notification = new ENotificationImpl(this, Notification.SET, MiniJavaPackage.FRAME__CHILD_FRAME, oldChildFrame, newChildFrame);
			if (msgs == null)
				msgs = notification;
			else
				msgs.add(notification);
		}
		return msgs;
	}

	public void setChildFrame(Frame newChildFrame) {
		if (newChildFrame != childFrame) {
			NotificationChain msgs = null;
			if (childFrame != null)
				msgs = ((InternalEObject) childFrame).eInverseRemove(this, MiniJavaPackage.FRAME__PARENT_FRAME, Frame.class, msgs);
			if (newChildFrame != null)
				msgs = ((InternalEObject) newChildFrame).eInverseAdd(this, MiniJavaPackage.FRAME__PARENT_FRAME, Frame.class, msgs);
			msgs = basicSetChildFrame(newChildFrame, msgs);
			if (msgs != null)
				msgs.dispatch();
		} else if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, MiniJavaPackage.FRAME__CHILD_FRAME, newChildFrame, newChildFrame));
	}

	public Frame getParentFrame() {
		if (eContainerFeatureID() != MiniJavaPackage.FRAME__PARENT_FRAME)
			return null;
		return (Frame) eInternalContainer();
	}

	public NotificationChain basicSetParentFrame(Frame newParentFrame, NotificationChain msgs) {
		msgs = eBasicSetContainer((InternalEObject) newParentFrame, MiniJavaPackage.FRAME__PARENT_FRAME, msgs);
		return msgs;
	}

	public void setParentFrame(Frame newParentFrame) {
		if (newParentFrame != eInternalContainer() || (eContainerFeatureID() != MiniJavaPackage.FRAME__PARENT_FRAME && newParentFrame != null)) {
			if (EcoreUtil.isAncestor(this, newParentFrame))
				throw new IllegalArgumentException("Recursive containment not allowed for " + toString());
			NotificationChain msgs = null;
			if (eInternalContainer() != null)
				msgs = eBasicRemoveFromContainer(msgs);
			if (newParentFrame != null)
				msgs = ((InternalEObject) newParentFrame).eInverseAdd(this, MiniJavaPackage.FRAME__CHILD_FRAME, Frame.class, msgs);
			msgs = basicSetParentFrame(newParentFrame, msgs);
			if (msgs != null)
				msgs.dispatch();
		} else if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, MiniJavaPackage.FRAME__PARENT_FRAME, newParentFrame, newParentFrame));
	}

	public Context getRootContext() {
		return rootContext;
	}

	public NotificationChain basicSetRootContext(Context newRootContext, NotificationChain msgs) {
		Context oldRootContext = rootContext;
		rootContext = newRootContext;
		if (eNotificationRequired()) {
			ENotificationImpl notification = new ENotificationImpl(this, Notification.SET, MiniJavaPackage.FRAME__ROOT_CONTEXT, oldRootContext, newRootContext);
			if (msgs == null)
				msgs = notification;
			else
				msgs.add(notification);
		}
		return msgs;
	}

	public void setRootContext(Context newRootContext) {
		if (newRootContext != rootContext) {
			NotificationChain msgs = null;
			if (rootContext != null)
				msgs = ((InternalEObject) rootContext).eInverseRemove(this, EOPPOSITE_FEATURE_BASE - MiniJavaPackage.FRAME__ROOT_CONTEXT, null, msgs);
			if (newRootContext != null)
				msgs = ((InternalEObject) newRootContext).eInverseAdd(this, EOPPOSITE_FEATURE_BASE - MiniJavaPackage.FRAME__ROOT_CONTEXT, null, msgs);
			msgs = basicSetRootContext(newRootContext, msgs);
			if (msgs != null)
				msgs.dispatch();
		} else if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, MiniJavaPackage.FRAME__ROOT_CONTEXT, newRootContext, newRootContext));
	}

	public Value getReturnValue() {
		return returnValue;
	}

	public NotificationChain basicSetReturnValue(Value newReturnValue, NotificationChain msgs) {
		Value oldReturnValue = returnValue;
		returnValue = newReturnValue;
		if (eNotificationRequired()) {
			ENotificationImpl notification = new ENotificationImpl(this, Notification.SET, MiniJavaPackage.FRAME__RETURN_VALUE, oldReturnValue, newReturnValue);
			if (msgs == null)
				msgs = notification;
			else
				msgs.add(notification);
		}
		return msgs;
	}

	public void setReturnValue(Value newReturnValue) {
		if (newReturnValue != returnValue) {
			NotificationChain msgs = null;
			if (returnValue != null)
				msgs = ((InternalEObject) returnValue).eInverseRemove(this, EOPPOSITE_FEATURE_BASE - MiniJavaPackage.FRAME__RETURN_VALUE, null, msgs);
			if (newReturnValue != null)
				msgs = ((InternalEObject) newReturnValue).eInverseAdd(this, EOPPOSITE_FEATURE_BASE - MiniJavaPackage.FRAME__RETURN_VALUE, null, msgs);
			msgs = basicSetReturnValue(newReturnValue, msgs);
			if (msgs != null)
				msgs.dispatch();
		} else if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, MiniJavaPackage.FRAME__RETURN_VALUE, newReturnValue, newReturnValue));
	}

	@Override
	public NotificationChain eInverseAdd(InternalEObject otherEnd, int featureID,
			NotificationChain msgs) {
		switch (featureID) {
			case MiniJavaPackage.FRAME__CHILD_FRAME :
				if (childFrame != null)
					msgs = ((InternalEObject) childFrame).eInverseRemove(this, EOPPOSITE_FEATURE_BASE - MiniJavaPackage.FRAME__CHILD_FRAME, null, msgs);
				return basicSetChildFrame((Frame) otherEnd, msgs);
			case MiniJavaPackage.FRAME__PARENT_FRAME :
				if (eInternalContainer() != null)
					msgs = eBasicRemoveFromContainer(msgs);
				return basicSetParentFrame((Frame) otherEnd, msgs);
		}
		return super.eInverseAdd(otherEnd, featureID, msgs);
	}

	@Override
	public NotificationChain eInverseRemove(InternalEObject otherEnd, int featureID,
			NotificationChain msgs) {
		switch (featureID) {
			case MiniJavaPackage.FRAME__CALL :
				return basicSetCall(null, msgs);
			case MiniJavaPackage.FRAME__CHILD_FRAME :
				return basicSetChildFrame(null, msgs);
			case MiniJavaPackage.FRAME__PARENT_FRAME :
				return basicSetParentFrame(null, msgs);
			case MiniJavaPackage.FRAME__ROOT_CONTEXT :
				return basicSetRootContext(null, msgs);
			case MiniJavaPackage.FRAME__RETURN_VALUE :
				return basicSetReturnValue(null, msgs);
		}
		return super.eInverseRemove(otherEnd, featureID, msgs);
	}

	@Override
	public NotificationChain eBasicRemoveFromContainerFeature(NotificationChain msgs) {
		switch (eContainerFeatureID()) {
			case MiniJavaPackage.FRAME__PARENT_FRAME :
				return eInternalContainer().eInverseRemove(this, MiniJavaPackage.FRAME__CHILD_FRAME, Frame.class, msgs);
		}
		return super.eBasicRemoveFromContainerFeature(msgs);
	}

	@Override
	public Object eGet(int featureID, boolean resolve, boolean coreType) {
		switch (featureID) {
			case MiniJavaPackage.FRAME__CALL :
				return getCall();
			case MiniJavaPackage.FRAME__INSTANCE :
				if (resolve)
					return getInstance();
				return basicGetInstance();
			case MiniJavaPackage.FRAME__CHILD_FRAME :
				return getChildFrame();
			case MiniJavaPackage.FRAME__PARENT_FRAME :
				return getParentFrame();
			case MiniJavaPackage.FRAME__ROOT_CONTEXT :
				return getRootContext();
			case MiniJavaPackage.FRAME__RETURN_VALUE :
				return getReturnValue();
		}
		return super.eGet(featureID, resolve, coreType);
	}

	@Override
	public void eSet(int featureID, Object newValue) {
		switch (featureID) {
			case MiniJavaPackage.FRAME__CALL :
				setCall((Call) newValue);
				return;
			case MiniJavaPackage.FRAME__INSTANCE :
				setInstance((ObjectInstance) newValue);
				return;
			case MiniJavaPackage.FRAME__CHILD_FRAME :
				setChildFrame((Frame) newValue);
				return;
			case MiniJavaPackage.FRAME__PARENT_FRAME :
				setParentFrame((Frame) newValue);
				return;
			case MiniJavaPackage.FRAME__ROOT_CONTEXT :
				setRootContext((Context) newValue);
				return;
			case MiniJavaPackage.FRAME__RETURN_VALUE :
				setReturnValue((Value) newValue);
				return;
		}
		super.eSet(featureID, newValue);
	}

	@Override
	public void eUnset(int featureID) {
		switch (featureID) {
			case MiniJavaPackage.FRAME__CALL :
				setCall((Call) null);
				return;
			case MiniJavaPackage.FRAME__INSTANCE :
				setInstance((ObjectInstance) null);
				return;
			case MiniJavaPackage.FRAME__CHILD_FRAME :
				setChildFrame((Frame) null);
				return;
			case MiniJavaPackage.FRAME__PARENT_FRAME :
				setParentFrame((Frame) null);
				return;
			case MiniJavaPackage.FRAME__ROOT_CONTEXT :
				setRootContext((Context) null);
				return;
			case MiniJavaPackage.FRAME__RETURN_VALUE :
				setReturnValue((Value) null);
				return;
		}
		super.eUnset(featureID);
	}

	@Override
	public boolean eIsSet(int featureID) {
		switch (featureID) {
			case MiniJavaPackage.FRAME__CALL :
				return call != null;
			case MiniJavaPackage.FRAME__INSTANCE :
				return instance != null;
			case MiniJavaPackage.FRAME__CHILD_FRAME :
				return childFrame != null;
			case MiniJavaPackage.FRAME__PARENT_FRAME :
				return getParentFrame() != null;
			case MiniJavaPackage.FRAME__ROOT_CONTEXT :
				return rootContext != null;
			case MiniJavaPackage.FRAME__RETURN_VALUE :
				return returnValue != null;
		}
		return super.eIsSet(featureID);
	}

	@Override
	public Object accept(VisitorInterface visitor) {
		return visitor.visitminiJava__Frame(this);
	}

	@Override
	public void attach(IDynamicModule dm) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void detach(IDynamicModule dm) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void notifyDynamicModules() {
		// TODO Auto-generated method stub
		
	}
}
