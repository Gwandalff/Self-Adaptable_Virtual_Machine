package miniJava.visitor.miniJava.impl;

import java.lang.Object;
import java.lang.Override;
import java.util.Collection;
import java.util.HashSet;
import java.util.Set;

import miniJava.visitor.miniJava.ArrayInstance;
import miniJava.visitor.miniJava.Context;
import miniJava.visitor.miniJava.Frame;
import miniJava.visitor.miniJava.MiniJavaPackage;
import miniJava.visitor.miniJava.ObjectInstance;
import miniJava.visitor.miniJava.OutputStream;
import miniJava.visitor.miniJava.State;
import org.eclipse.emf.common.notify.Notification;
import org.eclipse.emf.common.notify.NotificationChain;
import org.eclipse.emf.common.util.EList;
import org.eclipse.emf.ecore.EClass;
import org.eclipse.emf.ecore.InternalEObject;
import org.eclipse.emf.ecore.impl.ENotificationImpl;
import org.eclipse.emf.ecore.impl.MinimalEObjectImpl;
import org.eclipse.emf.ecore.util.EObjectContainmentEList;
import org.eclipse.emf.ecore.util.InternalEList;

import visitor.IDynamicModule;
import visitor.VisitorInterface;

public class StateImpl extends MinimalEObjectImpl.Container implements State {
	protected Set<IDynamicModule> modules;
	
	protected Frame rootFrame;

	protected EList<ObjectInstance> objectsHeap;

	protected OutputStream outputStream;

	protected EList<ArrayInstance> arraysHeap;

	protected Context contextCache;

	protected Frame frameCache;

	protected StateImpl() {
		super();
		modules = new HashSet<IDynamicModule>();
	}

	@Override
	protected EClass eStaticClass() {
		return MiniJavaPackage.Literals.STATE;
	}

	public Frame getRootFrame() {
		return rootFrame;
	}

	public NotificationChain basicSetRootFrame(Frame newRootFrame, NotificationChain msgs) {
		Frame oldRootFrame = rootFrame;
		rootFrame = newRootFrame;
		if (eNotificationRequired()) {
			ENotificationImpl notification = new ENotificationImpl(this, Notification.SET, MiniJavaPackage.STATE__ROOT_FRAME, oldRootFrame, newRootFrame);
			if (msgs == null)
				msgs = notification;
			else
				msgs.add(notification);
		}
		return msgs;
	}

	public void setRootFrame(Frame newRootFrame) {
		if (newRootFrame != rootFrame) {
			NotificationChain msgs = null;
			if (rootFrame != null)
				msgs = ((InternalEObject) rootFrame).eInverseRemove(this, EOPPOSITE_FEATURE_BASE - MiniJavaPackage.STATE__ROOT_FRAME, null, msgs);
			if (newRootFrame != null)
				msgs = ((InternalEObject) newRootFrame).eInverseAdd(this, EOPPOSITE_FEATURE_BASE - MiniJavaPackage.STATE__ROOT_FRAME, null, msgs);
			msgs = basicSetRootFrame(newRootFrame, msgs);
			if (msgs != null)
				msgs.dispatch();
		} else if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, MiniJavaPackage.STATE__ROOT_FRAME, newRootFrame, newRootFrame));
	}

	public EList<ObjectInstance> getObjectsHeap() {
		if (objectsHeap == null) {
			objectsHeap = new EObjectContainmentEList<ObjectInstance>(ObjectInstance.class, this, MiniJavaPackage.STATE__OBJECTS_HEAP);
		}
		return objectsHeap;
	}

	public OutputStream getOutputStream() {
		return outputStream;
	}

	public NotificationChain basicSetOutputStream(OutputStream newOutputStream,
			NotificationChain msgs) {
		OutputStream oldOutputStream = outputStream;
		outputStream = newOutputStream;
		if (eNotificationRequired()) {
			ENotificationImpl notification = new ENotificationImpl(this, Notification.SET, MiniJavaPackage.STATE__OUTPUT_STREAM, oldOutputStream, newOutputStream);
			if (msgs == null)
				msgs = notification;
			else
				msgs.add(notification);
		}
		return msgs;
	}

	public void setOutputStream(OutputStream newOutputStream) {
		if (newOutputStream != outputStream) {
			NotificationChain msgs = null;
			if (outputStream != null)
				msgs = ((InternalEObject) outputStream).eInverseRemove(this, EOPPOSITE_FEATURE_BASE - MiniJavaPackage.STATE__OUTPUT_STREAM, null, msgs);
			if (newOutputStream != null)
				msgs = ((InternalEObject) newOutputStream).eInverseAdd(this, EOPPOSITE_FEATURE_BASE - MiniJavaPackage.STATE__OUTPUT_STREAM, null, msgs);
			msgs = basicSetOutputStream(newOutputStream, msgs);
			if (msgs != null)
				msgs.dispatch();
		} else if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, MiniJavaPackage.STATE__OUTPUT_STREAM, newOutputStream, newOutputStream));
	}

	public EList<ArrayInstance> getArraysHeap() {
		if (arraysHeap == null) {
			arraysHeap = new EObjectContainmentEList<ArrayInstance>(ArrayInstance.class, this, MiniJavaPackage.STATE__ARRAYS_HEAP);
		}
		return arraysHeap;
	}

	public Context getContextCache() {
		if (contextCache != null && contextCache.eIsProxy()) {
			InternalEObject oldContextCache = (InternalEObject) contextCache;
			contextCache = (Context) eResolveProxy(oldContextCache);
			if (contextCache != oldContextCache) {
				if (eNotificationRequired())
					eNotify(new ENotificationImpl(this, Notification.RESOLVE, MiniJavaPackage.STATE__CONTEXT_CACHE, oldContextCache, contextCache));
			}
		}
		return contextCache;
	}

	public Context basicGetContextCache() {
		return contextCache;
	}

	public void setContextCache(Context newContextCache) {
		Context oldContextCache = contextCache;
		contextCache = newContextCache;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, MiniJavaPackage.STATE__CONTEXT_CACHE, oldContextCache, contextCache));
	}

	public Frame getFrameCache() {
		if (frameCache != null && frameCache.eIsProxy()) {
			InternalEObject oldFrameCache = (InternalEObject) frameCache;
			frameCache = (Frame) eResolveProxy(oldFrameCache);
			if (frameCache != oldFrameCache) {
				if (eNotificationRequired())
					eNotify(new ENotificationImpl(this, Notification.RESOLVE, MiniJavaPackage.STATE__FRAME_CACHE, oldFrameCache, frameCache));
			}
		}
		return frameCache;
	}

	public Frame basicGetFrameCache() {
		return frameCache;
	}

	public void setFrameCache(Frame newFrameCache) {
		Frame oldFrameCache = frameCache;
		frameCache = newFrameCache;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, MiniJavaPackage.STATE__FRAME_CACHE, oldFrameCache, frameCache));
	}

	@Override
	public NotificationChain eInverseRemove(InternalEObject otherEnd, int featureID,
			NotificationChain msgs) {
		switch (featureID) {
			case MiniJavaPackage.STATE__ROOT_FRAME :
				return basicSetRootFrame(null, msgs);
			case MiniJavaPackage.STATE__OBJECTS_HEAP :
				return ((InternalEList<?>) getObjectsHeap()).basicRemove(otherEnd, msgs);
			case MiniJavaPackage.STATE__OUTPUT_STREAM :
				return basicSetOutputStream(null, msgs);
			case MiniJavaPackage.STATE__ARRAYS_HEAP :
				return ((InternalEList<?>) getArraysHeap()).basicRemove(otherEnd, msgs);
		}
		return super.eInverseRemove(otherEnd, featureID, msgs);
	}

	@Override
	public Object eGet(int featureID, boolean resolve, boolean coreType) {
		switch (featureID) {
			case MiniJavaPackage.STATE__ROOT_FRAME :
				return getRootFrame();
			case MiniJavaPackage.STATE__OBJECTS_HEAP :
				return getObjectsHeap();
			case MiniJavaPackage.STATE__OUTPUT_STREAM :
				return getOutputStream();
			case MiniJavaPackage.STATE__ARRAYS_HEAP :
				return getArraysHeap();
			case MiniJavaPackage.STATE__CONTEXT_CACHE :
				if (resolve)
					return getContextCache();
				return basicGetContextCache();
			case MiniJavaPackage.STATE__FRAME_CACHE :
				if (resolve)
					return getFrameCache();
				return basicGetFrameCache();
		}
		return super.eGet(featureID, resolve, coreType);
	}

	@Override
	public void eSet(int featureID, Object newValue) {
		switch (featureID) {
			case MiniJavaPackage.STATE__ROOT_FRAME :
				setRootFrame((Frame) newValue);
				return;
			case MiniJavaPackage.STATE__OBJECTS_HEAP :
				getObjectsHeap().clear();
				getObjectsHeap().addAll((Collection<? extends ObjectInstance>) newValue);
				return;
			case MiniJavaPackage.STATE__OUTPUT_STREAM :
				setOutputStream((OutputStream) newValue);
				return;
			case MiniJavaPackage.STATE__ARRAYS_HEAP :
				getArraysHeap().clear();
				getArraysHeap().addAll((Collection<? extends ArrayInstance>) newValue);
				return;
			case MiniJavaPackage.STATE__CONTEXT_CACHE :
				setContextCache((Context) newValue);
				return;
			case MiniJavaPackage.STATE__FRAME_CACHE :
				setFrameCache((Frame) newValue);
				return;
		}
		super.eSet(featureID, newValue);
	}

	@Override
	public void eUnset(int featureID) {
		switch (featureID) {
			case MiniJavaPackage.STATE__ROOT_FRAME :
				setRootFrame((Frame) null);
				return;
			case MiniJavaPackage.STATE__OBJECTS_HEAP :
				getObjectsHeap().clear();
				return;
			case MiniJavaPackage.STATE__OUTPUT_STREAM :
				setOutputStream((OutputStream) null);
				return;
			case MiniJavaPackage.STATE__ARRAYS_HEAP :
				getArraysHeap().clear();
				return;
			case MiniJavaPackage.STATE__CONTEXT_CACHE :
				setContextCache((Context) null);
				return;
			case MiniJavaPackage.STATE__FRAME_CACHE :
				setFrameCache((Frame) null);
				return;
		}
		super.eUnset(featureID);
	}

	@Override
	public boolean eIsSet(int featureID) {
		switch (featureID) {
			case MiniJavaPackage.STATE__ROOT_FRAME :
				return rootFrame != null;
			case MiniJavaPackage.STATE__OBJECTS_HEAP :
				return objectsHeap != null && !objectsHeap.isEmpty();
			case MiniJavaPackage.STATE__OUTPUT_STREAM :
				return outputStream != null;
			case MiniJavaPackage.STATE__ARRAYS_HEAP :
				return arraysHeap != null && !arraysHeap.isEmpty();
			case MiniJavaPackage.STATE__CONTEXT_CACHE :
				return contextCache != null;
			case MiniJavaPackage.STATE__FRAME_CACHE :
				return frameCache != null;
		}
		return super.eIsSet(featureID);
	}

	@Override
	public Object accept(VisitorInterface visitor) {
		return visitor.visitminiJava__State(this);
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
	public boolean notifyDynamicModulesBefore() {
		boolean out = true;
		for (IDynamicModule dm : modules) {
			out = out && dm.updateBefore(this);
		}
		return out;
	}

	@Override
	public void notifyDynamicModulesAfter() {
		for (IDynamicModule dm : modules) {
			dm.updateAfter(this);
		}
	}
}
