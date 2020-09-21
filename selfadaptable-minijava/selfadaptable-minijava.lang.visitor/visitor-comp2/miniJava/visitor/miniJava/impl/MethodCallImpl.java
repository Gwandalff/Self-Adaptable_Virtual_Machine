package miniJava.visitor.miniJava.impl;

import java.lang.Object;
import java.lang.Override;
import java.util.Collection;
import java.util.HashSet;
import java.util.Set;

import miniJava.visitor.miniJava.Expression;
import miniJava.visitor.miniJava.Method;
import miniJava.visitor.miniJava.MethodCall;
import miniJava.visitor.miniJava.MiniJavaPackage;
import org.eclipse.emf.common.notify.Notification;
import org.eclipse.emf.common.notify.NotificationChain;
import org.eclipse.emf.common.util.EList;
import org.eclipse.emf.ecore.EClass;
import org.eclipse.emf.ecore.InternalEObject;
import org.eclipse.emf.ecore.impl.ENotificationImpl;
import org.eclipse.emf.ecore.util.EObjectContainmentEList;
import org.eclipse.emf.ecore.util.InternalEList;

import visitor.IDynamicModule;
import visitor.VisitorInterface;

public class MethodCallImpl extends ExpressionImpl implements MethodCall {
	protected Set<IDynamicModule> modules;
	
	protected Expression receiver;

	protected Method method;

	protected EList<Expression> args;

	protected MethodCallImpl() {
		super();
		modules = new HashSet<IDynamicModule>();
	}

	@Override
	protected EClass eStaticClass() {
		return MiniJavaPackage.Literals.METHOD_CALL;
	}

	public Expression getReceiver() {
		return receiver;
	}

	public NotificationChain basicSetReceiver(Expression newReceiver, NotificationChain msgs) {
		Expression oldReceiver = receiver;
		receiver = newReceiver;
		if (eNotificationRequired()) {
			ENotificationImpl notification = new ENotificationImpl(this, Notification.SET, MiniJavaPackage.METHOD_CALL__RECEIVER, oldReceiver, newReceiver);
			if (msgs == null)
				msgs = notification;
			else
				msgs.add(notification);
		}
		return msgs;
	}

	public void setReceiver(Expression newReceiver) {
		if (newReceiver != receiver) {
			NotificationChain msgs = null;
			if (receiver != null)
				msgs = ((InternalEObject) receiver).eInverseRemove(this, EOPPOSITE_FEATURE_BASE - MiniJavaPackage.METHOD_CALL__RECEIVER, null, msgs);
			if (newReceiver != null)
				msgs = ((InternalEObject) newReceiver).eInverseAdd(this, EOPPOSITE_FEATURE_BASE - MiniJavaPackage.METHOD_CALL__RECEIVER, null, msgs);
			msgs = basicSetReceiver(newReceiver, msgs);
			if (msgs != null)
				msgs.dispatch();
		} else if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, MiniJavaPackage.METHOD_CALL__RECEIVER, newReceiver, newReceiver));
	}

	public Method getMethod() {
		if (method != null && method.eIsProxy()) {
			InternalEObject oldMethod = (InternalEObject) method;
			method = (Method) eResolveProxy(oldMethod);
			if (method != oldMethod) {
				if (eNotificationRequired())
					eNotify(new ENotificationImpl(this, Notification.RESOLVE, MiniJavaPackage.METHOD_CALL__METHOD, oldMethod, method));
			}
		}
		return method;
	}

	public Method basicGetMethod() {
		return method;
	}

	public void setMethod(Method newMethod) {
		Method oldMethod = method;
		method = newMethod;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, MiniJavaPackage.METHOD_CALL__METHOD, oldMethod, method));
	}

	public EList<Expression> getArgs() {
		if (args == null) {
			args = new EObjectContainmentEList<Expression>(Expression.class, this, MiniJavaPackage.METHOD_CALL__ARGS);
		}
		return args;
	}

	@Override
	public NotificationChain eInverseRemove(InternalEObject otherEnd, int featureID,
			NotificationChain msgs) {
		switch (featureID) {
			case MiniJavaPackage.METHOD_CALL__RECEIVER :
				return basicSetReceiver(null, msgs);
			case MiniJavaPackage.METHOD_CALL__ARGS :
				return ((InternalEList<?>) getArgs()).basicRemove(otherEnd, msgs);
		}
		return super.eInverseRemove(otherEnd, featureID, msgs);
	}

	@Override
	public Object eGet(int featureID, boolean resolve, boolean coreType) {
		switch (featureID) {
			case MiniJavaPackage.METHOD_CALL__RECEIVER :
				return getReceiver();
			case MiniJavaPackage.METHOD_CALL__METHOD :
				if (resolve)
					return getMethod();
				return basicGetMethod();
			case MiniJavaPackage.METHOD_CALL__ARGS :
				return getArgs();
		}
		return super.eGet(featureID, resolve, coreType);
	}

	@Override
	public void eSet(int featureID, Object newValue) {
		switch (featureID) {
			case MiniJavaPackage.METHOD_CALL__RECEIVER :
				setReceiver((Expression) newValue);
				return;
			case MiniJavaPackage.METHOD_CALL__METHOD :
				setMethod((Method) newValue);
				return;
			case MiniJavaPackage.METHOD_CALL__ARGS :
				getArgs().clear();
				getArgs().addAll((Collection<? extends Expression>) newValue);
				return;
		}
		super.eSet(featureID, newValue);
	}

	@Override
	public void eUnset(int featureID) {
		switch (featureID) {
			case MiniJavaPackage.METHOD_CALL__RECEIVER :
				setReceiver((Expression) null);
				return;
			case MiniJavaPackage.METHOD_CALL__METHOD :
				setMethod((Method) null);
				return;
			case MiniJavaPackage.METHOD_CALL__ARGS :
				getArgs().clear();
				return;
		}
		super.eUnset(featureID);
	}

	@Override
	public boolean eIsSet(int featureID) {
		switch (featureID) {
			case MiniJavaPackage.METHOD_CALL__RECEIVER :
				return receiver != null;
			case MiniJavaPackage.METHOD_CALL__METHOD :
				return method != null;
			case MiniJavaPackage.METHOD_CALL__ARGS :
				return args != null && !args.isEmpty();
		}
		return super.eIsSet(featureID);
	}

	@Override
	public Object accept(VisitorInterface visitor) {
		return visitor.visitminiJava__MethodCall(this);
	}
}
